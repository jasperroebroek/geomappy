class Bounds:
    """
    Object containing geographical bounds to interact with maps

    Parameters
    ----------
    x0, x1, y0, y1 : float
        Longitude and latitude values
    check_bounds : bool, optional
        If True, the longitude and latitude values are checked for correctness

    Notes
    -----
    This object can be interacted with by either asking for the specifc coordinates:
    >>> Bounds(0,10,0,10).x1
    >>> 10

    Or you can obtain the values in this order (x0,x1,y0,y1) in the iterator interface
    >>> print(*Bounds(0,10,0,10))
    >>> 0 10 0 10

    """
    def __init__(self, x0, x1, y0, y1, check_bounds=True):
        if not isinstance(check_bounds, bool):
            raise TypeError("check_bounds is a boolean parameter")
        else:
            self._check_bounds = check_bounds

        self._x0 = None
        self._x1 = None
        self._y0 = None
        self._y1 = None

        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1

    def set_x0(self, x0):
        if x0 < -180 and self._check_bounds:
            raise ValueError("x0 value out of bounds")
        else:
            self._x0 = x0

    def get_x0(self):
        return self._x0

    x0 = property(get_x0, set_x0)

    def set_x1(self, x1):
        if x1 > 180 and self._check_bounds:
            raise ValueError("x1 value is out of bounds")
        else:
            self._x1 = x1

    def get_x1(self):
        return self._x1

    x1 = property(get_x1, set_x1)

    def set_y0(self, y0):
        if y0 < -180 and self._check_bounds:
            raise ValueError("y0 value out of bounds")
        else:
            self._y0 = y0

    def get_y0(self):
        return self._y0

    y0 = property(get_y0, set_y0)

    def set_y1(self, y1):
        if y1 > 180 and self._check_bounds:
            raise ValueError("y1 value is out of bounds")
        else:
            self._y1 = y1

    def get_y1(self):
        return self._y1

    y1 = property(get_y1, set_y1)

    def __str__(self):
        return f"Bounds[x0={self.x0}, x1={self.x1}, y0={self.y0}, y1={self.y1}]"

    def __iter__(self):
        return iter((self.x0, self.x1, self.y0, self.y1))
