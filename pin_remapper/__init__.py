import pcbnew
import wx
from pcbnew import ActionPlugin
import math

class NetGeometryExtractor(ActionPlugin):

    def defaults(self):
        self.name = "Net Geometry Extractor"
        self.category = "Net Utilities"
        self.description = "Extracts geometry grouped by net (tracks, pads, zones)"
        self.show_toolbar_button = True

        self.icon_file_name = ""  # Add icon path if needed

    def Run(self):

        # Grab all the pins in the design
        pins = get_selected_pins()
        if len(pins) == 0:
            self.dialog("Need to select a component to remap pins on")
            return
       
        pins_to_filter = [(i, str(pad.GetPadName()), str(pad.GetNet().GetNetname())) for i, pad in enumerate(pins)]

        dlg = RegexDialog()
        dlg.set_pins(pins_to_filter)
        result = dlg.ShowModal()
        if result != wx.ID_OK:
            self.dialog("No pins selected to remap")
            return
        pins = [pins[i] for i in dlg.GetPinIndices()]

        dlg.Destroy()


        board = pcbnew.GetBoard()
       
        count = 0

        while True:
            names = [pin.GetNet().GetNetname() for pin in pins]
            nets = [pin.GetNet() for pin in pins]
            pin_geometries = []
            other_geometries = []
            distances = [[float('inf') for _ in names] for _ in names]

            for pin in pins:
                #self.__print(pin)
                pin_geo, other_geo = split_geometry_by_pin(board, pin)
                pin_geometries.append(pin_geo)
                other_geometries.append(other_geo)
                #self.__print("Finished")
            
            for i, pin_geom in enumerate(pin_geometries):
                for j, other_geom in enumerate(other_geometries):
                    if i > j:
                        continue
                    dist = closest_distance_between_geoms(pin_geom, other_geom)
                    distances[i][j] = dist
                    distances[j][i] = dist
                    #self.__print(f"{names[i]} -> {names[j]} = {dist}")
            #self.__print("Starting")

            solution = hungarian_algorithm(distances)

            if solution == list(range(0, len(solution))):
                #No work to do!
                return

            solution = {solution[i]:i for i in solution}
            self.__print(solution)
      
            for i, geometry in enumerate(pin_geometries):
                for obj in geometry:
                    obj["object"].SetNet(nets[solution[i]])

            pcbnew.Refresh()

            count += 1

            if count > 10:
                return


    def dialog(self, message):
        dlg = wx.MessageDialog(None, message, "Net Geometry Summary", wx.OK)
        dlg.ShowModal()
        dlg.Destroy()


# ----- And now, for something hungarian ---- #

def hungarian_algorithm(cost_matrix):
    import copy
    n = len(cost_matrix)
    cost = copy.deepcopy(cost_matrix)

    u = [0] * n
    v = [0] * n
    mark_indices = [-1] * n

    for i in range(n):
        links = [-1] * n
        mins = [float('inf')] * n
        visited = [False] * n
        marked_i = i
        marked_j = -1
        j = 0

        while True:
            j = -1
            for j1 in range(n):
                if not visited[j1]:
                    cur = cost[marked_i][j1] - u[marked_i] - v[j1]
                    if cur < mins[j1]:
                        mins[j1] = cur
                        links[j1] = marked_j
                    if j == -1 or mins[j1] < mins[j]:
                        j = j1
            delta = mins[j]
            for j1 in range(n):
                if visited[j1]:
                    u[mark_indices[j1]] += delta
                    v[j1] -= delta
                else:
                    mins[j1] -= delta
            u[i] += delta
            visited[j] = True
            marked_j = j
            marked_i = mark_indices[j]
            if marked_i == -1:
                break

        # Augmenting path
        while True:
            if links[j] != -1:
                mark_indices[j] = mark_indices[links[j]]
                j = links[j]
            else:
                break
        mark_indices[j] = i

    # Build the result: index i in from_list is assigned to mark_indices.index(i)
    result = [-1] * n
    for j in range(n):
        if mark_indices[j] != -1:
            result[mark_indices[j]] = j
    return result

# ---- Something ---- #
def distance_points(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def distance_circle_circle(c1, c2):
    center_dist = distance_points(c1.center, c2.center)
    dist = max(0, center_dist - c1.radius - c2.radius)
    return dist

def distance_rect_rect(r1, r2):
    left = max(r1.bbox_min.x, r2.bbox_min.x)
    right = min(r1.bbox_max.x, r2.bbox_max.x)
    bottom = max(r1.bbox_min.y, r2.bbox_min.y)
    top = min(r1.bbox_max.y, r2.bbox_max.y)

    if right >= left and top >= bottom:
        return 0  # overlapping

    dx = max(0, max(r1.bbox_min.x, r2.bbox_min.x) - min(r1.bbox_max.x, r2.bbox_max.x))
    dy = max(0, max(r1.bbox_min.y, r2.bbox_min.y) - min(r1.bbox_max.y, r2.bbox_max.y))
    return math.hypot(dx, dy)

def distance_rect_circle(rect, circle):
    # Clamp circle center to rect bounds
    cx, cy = circle.center.x, circle.center.y
    closest_x = min(max(cx, rect.bbox_min.x), rect.bbox_max.x)
    closest_y = min(max(cy, rect.bbox_min.y), rect.bbox_max.y)
    dist = distance_points(circle.center, pcbnew.wxPoint(closest_x, closest_y)) - circle.radius
    return max(0, dist)

def closest_distance_between_geoms(list1, list2):
    min_distance = float('inf')

    for geom1 in list1:
        for shape_type1, shape1 in geom1["shapes"]:
            for geom2 in list2:
                for shape_type2, shape2 in geom2["shapes"]:
                    dist = None

                    if shape_type1 == "circle" and shape_type2 == "circle":
                        dist = distance_circle_circle(shape1, shape2)
                    elif shape_type1 == "rect" and shape_type2 == "rect":
                        dist = distance_rect_rect(shape1, shape2)
                    elif shape_type1 == "rect" and shape_type2 == "circle":
                        dist = distance_rect_circle(shape1, shape2)
                    elif shape_type1 == "circle" and shape_type2 == "rect":
                        dist = distance_rect_circle(shape2, shape1)
                    else:
                        # Unknown shape types - skip or handle as needed
                        continue

                    if dist is not None and dist < min_distance:
                        min_distance = dist

    return min_distance

def vector_length(v):
    return math.hypot(v.x, v.y)

def vector_sub(v1, v2):
    return pcbnew.wxPoint(v1.x - v2.x, v1.y - v2.y)

def vector_add(v1, v2):
    return pcbnew.wxPoint(v1.x + v2.x, v1.y + v2.y)

def vector_dot(v1, v2):
    return v1.x * v2.x + v1.y * v2.y

def point_distance(p1, p2):
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

# --- Geometry primitives and intersection checks ---

class Circle:
    def __init__(self, center: pcbnew.wxPoint, radius: float):
        self.center = center
        self.radius = radius

class Rectangle:
    def __init__(self, start: pcbnew.wxPoint, end: pcbnew.wxPoint, width: float):
        # Axis-aligned bounding box around the track segment, inflated by half width
        # We'll build a rectangle as center line + width, but intersection check uses AABB for simplicity

        # Store points and width
        self.start = start
        self.end = end
        self.width = width

        # Calculate oriented rectangle corners (optional: for now just bounding box)
        min_x = min(start.x, end.x) - width // 2
        max_x = max(start.x, end.x) + width // 2
        min_y = min(start.y, end.y) - width // 2
        max_y = max(start.y, end.y) + width // 2
        self.bbox_min = pcbnew.wxPoint(min_x, min_y)
        self.bbox_max = pcbnew.wxPoint(max_x, max_y)

def rects_intersect(rect1: Rectangle, rect2: Rectangle):
    # Simple AABB overlap test
    return not (rect1.bbox_max.x < rect2.bbox_min.x or
                rect1.bbox_min.x > rect2.bbox_max.x or
                rect1.bbox_max.y < rect2.bbox_min.y or
                rect1.bbox_min.y > rect2.bbox_max.y)

def circle_circle_intersect(c1: Circle, c2: Circle):
    dist = point_distance(c1.center, c2.center)
    return dist <= (c1.radius + c2.radius)

def circle_rect_intersect(circle: Circle, rect: Rectangle):
    # Clamp circle center to rectangle bbox to find closest point
    cx = circle.center.x
    cy = circle.center.y
    closest_x = max(rect.bbox_min.x, min(cx, rect.bbox_max.x))
    closest_y = max(rect.bbox_min.y, min(cy, rect.bbox_max.y))

    closest_point = pcbnew.wxPoint(closest_x, closest_y)
    dist = point_distance(circle.center, closest_point)
    return dist <= circle.radius

def shapes_intersect(shape1, shape2):
    # shape is dict with keys: "type", and "geom" (Circle or Rectangle instance)
    t1, g1 = shape1["type"], shape1["geom"]
    t2, g2 = shape2["type"], shape2["geom"]

    if t1 == "circle" and t2 == "circle":
        return circle_circle_intersect(g1, g2)
    elif t1 == "rect" and t2 == "rect":
        return rects_intersect(g1, g2)
    elif t1 == "circle" and t2 == "rect":
        return circle_rect_intersect(g1, g2)
    elif t1 == "rect" and t2 == "circle":
        return circle_rect_intersect(g2, g1)
    else:
        # Unknown types, consider no intersection
        return False

# --- Convert PCB geometry to shapes ---

def track_to_shapes(track):
    start = track.GetStart()
    end = track.GetEnd()
    width = track.GetWidth()

    rect = Rectangle(start, end, width)
    radius = width / 2
    c_start = Circle(start, radius)
    c_end = Circle(end, radius)
    return {
        "type": "track",
        "object": track,
        "shapes": [
            ("rect", rect),
            ("circle", c_start),
            ("circle", c_end),
        ],
    }

def via_to_shape(via):
    center = via.GetStart()
    radius = via.GetWidth() / 2
    circle = Circle(center, radius)
    return {
        "type": "via",
        "object": via,
        "shapes": [("circle", circle)]
    }

def pad_to_shape(pad):
    center = pad.GetPosition()
    size_x, size_y = pad.GetSize()
    # Approximate pad as circle with radius = max(size_x, size_y)/2
    radius = max(size_x, size_y) / 2
    circle = Circle(center, radius)
    return {
        "type": "pad",
        "object": pad,
        "shapes": [("circle", circle)]
    }

# --- Intersection between two geometry items (each with multiple shapes) ---

def geom_items_intersect(item1, item2):
    # item1 and item2 have "shapes" = list of (type, shape)
    for type1, shape1 in item1["shapes"]:
        for type2, shape2 in item2["shapes"]:
            if shapes_intersect({"type": type1, "geom": shape1},
                                {"type": type2, "geom": shape2}):
                return True
    return False

# --- Extract all geometry of a net ---

def extract_net_geometry(board, net):
    """
    Extracts all geometry (tracks, vias, pads) on the given net.
    `net` is a NETINFO_ITEM instance.

    Returns a list of geometry dicts with keys:
    - "type": "track", "via", or "pad"
    - "object": the original pcbnew object
    - "shapes": list of shape primitives (rectangles or circles) for geometry intersection
    """

    target_net_name = net.GetNetname()
    geometry = []

    # Helper functions to build shapes
    def track_to_shapes(track):
        start = track.GetStart()
        end = track.GetEnd()
        width = track.GetWidth()

        # Rectangle bounding box + circles on ends for thickness
        radius = width / 2

        rect = Rectangle(start, end, width)
        c_start = Circle(start, radius)
        c_end = Circle(end, radius)

        return {
            "type": "track",
            "object": track,
            "shapes": [
                ("rect", rect),
                ("circle", c_start),
                ("circle", c_end),
            ],
        }

    def via_to_shape(via):
        center = via.GetStart()
        radius = via.GetWidth() / 2
        circle = Circle(center, radius)
        return {
            "type": "via",
            "object": via,
            "shapes": [("circle", circle)],
        }

    def pad_to_shape(pad):
        center = pad.GetPosition()
        size_x, size_y = pad.GetSize()
        radius = max(size_x, size_y) / 2
        circle = Circle(center, radius)
        return {
            "type": "pad",
            "object": pad,
            "shapes": [("circle", circle)],
        }

    # --- Classes Rectangle and Circle need to be defined in your scope as before ---

    # Iterate tracks
    for track in board.GetTracks():
        # Check net by name because GetNet() returns net code int, we can also get net name
        if track.GetNetname() != target_net_name:
            continue

        if isinstance(track, pcbnew.PCB_VIA):
            geometry.append(via_to_shape(track))
        else:
            geometry.append(track_to_shapes(track))

    # Iterate pads
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            if pad.GetNetname() == target_net_name:
                geometry.append(pad_to_shape(pad))

    return geometry

# --- Graph traversal to find connected geometry starting from pin geometry ---

def connected_geometry(geometry_list, start_geom):
    visited = set()
    to_visit = [start_geom]

    while to_visit:
        current = to_visit.pop()
        #print(current)
        visited.add(id(current["object"]))

        for geom in geometry_list:
            #print(f"Geom: {geom}")
            if id(geom["object"]) not in visited:
                if geom_items_intersect(current, geom):
                    to_visit.append(geom)

    return [g for g in geometry_list if id(g["object"]) in visited]

# --- Main split function ---

def split_geometry_by_pin(board, pad):
    
    net = pad.GetNet()
    if not net:
        return [], []

    all_geom = extract_net_geometry(board, net)
    start_geom = next((g for g in all_geom if g["object"] == pad), None)

    if not start_geom:
        # Pad not found? Return empty connected, all remaining
        return [], all_geom
    
    connected = connected_geometry(all_geom, start_geom)
    connected_set = set(id(g["object"]) for g in connected)
    remaining = [g for g in all_geom if id(g["object"]) not in connected_set]

    return connected, remaining

class RegexDialog(wx.Dialog):

    def set_pins(self, pin_list):
        self.pin_list = pin_list
        self.matched_pins = pin_list
        self.on_text_changed(None)

    def __init__(self):
        super().__init__(parent = wx.GetTopLevelWindows()[0])
        
        import wx.lib.scrolledpanel as scrolled


        self.SetTitle("Filter nets to remap")
        
        self.SetSize((400, 350))

        sizer = wx.BoxSizer(wx.VERTICAL)

        self.input = wx.TextCtrl(self)
        sizer.Add(self.input, 0, wx.EXPAND | wx.ALL, 10)

        self.feedback = wx.StaticText(self, label="")
        sizer.Add(self.feedback, 0, wx.EXPAND | wx.ALL, 10)

        self.scroll_panel = scrolled.ScrolledPanel(self, size=(-1, 200), style=wx.SIMPLE_BORDER)
        self.scroll_panel.SetupScrolling(scroll_x=False, scroll_y=True)
        self.scroll_sizer = wx.BoxSizer(wx.VERTICAL)
        self.scroll_panel.SetSizer(self.scroll_sizer)

        sizer.Add(self.scroll_panel, 0, wx.EXPAND | wx.ALL, 10)    

        # OK and Cancel buttons
        btn_sizer = wx.StdDialogButtonSizer()
        ok_btn = wx.Button(self, wx.ID_OK)
        cancel_btn = wx.Button(self, wx.ID_CANCEL)
        btn_sizer.AddButton(ok_btn)
        btn_sizer.AddButton(cancel_btn)
        btn_sizer.Realize()
        sizer.Add(btn_sizer, 0, wx.ALIGN_RIGHT | wx.ALL, 10)

        self.SetSizer(sizer)

        # Bind event for live text update
        self.input.Bind(wx.EVT_TEXT, self.on_text_changed)
        self.ok_btn = ok_btn

        # Initially disable OK until regex valid
        self.ok_btn.Enable(False)

    def on_text_changed(self, event):
        import re
        regex = self.input.GetValue()
        #self.__print(f"Regex is: %%{regex}%%")
        
        try:
            compiled = re.compile(regex)
        except re.error:
            compiled = None
        #self.__print(compiled)
        try:
            if compiled:
                self.matched_pins = []
                for (i, pad_name, net_name) in self.pin_list:
                    if compiled.match(net_name):
                        self.matched_pins.append((i, pad_name, net_name))
                    else:
                        pass#self.__print(f"Unmatched: {net_name}")
                if len(self.matched_pins) > 0:
                    self.feedback.SetLabel(f"{len(self.matched_pins)} Matching pins:")
                    self.populate_scroll([f"({pad_name}) - {net_name}" for _, pad_name, net_name in self.matched_pins])
                else:
                    self.feedback.SetLabel("No pins matched. Available pins:")
                    self.populate_scroll([f"({pad_name}) - {net_name}" for _, pad_name, net_name in self.pin_list])
                self.ok_btn.Enable(True)
            else:
                self.feedback.SetLabel("Invalid regex ❌")
                self.ok_btn.Enable(False)
        except Exception as e:
            import traceback
            tb = traceback.extract_tb(e.__traceback__)
            #self.__print(tb)
        
        if event:
            event.Skip()
    
    def GetPinIndices(self):
        return [i for i, _, _ in self.matched_pins]

    def populate_scroll(self, items):
        # Step 1: Destroy old children
        for child in self.scroll_panel.GetChildren():
            child.Destroy()

        # Step 2: Clear the sizer
        self.scroll_sizer.Clear(delete_windows=False)

        # Step 3: Add new items
        for item in items:
            txt = wx.StaticText(self.scroll_panel, label=item)
            self.scroll_sizer.Add(txt, 0, wx.ALL, 0)

        # Step 4: Refresh layout
        self.scroll_panel.Layout()
        self.scroll_panel.FitInside()

def get_selected_pins():
    selected_items = pcbnew.GetCurrentSelection()

    selected_footprints = [
        item for item in selected_items
        if isinstance(item, pcbnew.FOOTPRINT)
    ]

    if not selected_footprints:
        return [] 
    
    pad_list = [pad for fp in selected_footprints for pad in fp.Pads()]
    return pad_list


# Required to register the plugin
NetGeometryExtractor().register()
