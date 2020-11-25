

class ArrayIterator:
    """
    Provide consecutive windows by which a raster can be iterated over.
    The caller must be smart when choosing window shape, no checking is done here.
    """

    def __init__(self, raster, window_height, window_width):
        self.raster = raster
        self.window_height = window_height
        self.window_width = window_width
        self.current_window = ((0,window_height),(0,window_width))
        self.reached_end = False

    def go_to_next(self):
        # if not yet reached end of row
        if self.current_window[1][1]  < self.raster.width:
            self.current_window = (\
                self.current_window[0],\
                (self.current_window[1][1], self.current_window[1][1] + self.window_width)\
            )
        # if reached end of the row, but not end of table
        elif self.current_window[0][1] < self.raster.height:
            self.current_window = (\
                (self.current_window[0][1], self.current_window[0][1] + self.window_height),\
                (0, self.window_width)\
            )
        # if reached end of table
        else:
            self.reached_end = True

    def pop_window(self):
        current_window = self.current_window
        self.go_to_next()
        return current_window

    def has_reached_end(self):
        return self.reached_end

    def reset(self):
        self.current_window = ((0,self.window_height),(0,self.window_width))
        self.reached_end = False
