#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Progress bar
"""
import sys


def progress_bar(s, frac=True, line=True, width=20, prefix="", silent=True):
    """
    Taking either a percentage or a fraction and prints a progress bar that updates the line every time the function
    is called.
    
    Parameters
    ----------
    s : float
        Fraction or percentage of progress.
    frac : bool, optional
        If true, s is taken as a fraction, otherwise s is taken as a
        percentage which is the default
    line : bool, optional
        If True a line of '#' is printed to visually check progress.
        If False only the percentage is printed. The default is True.
    width : int, optional
        Width of the line of #. The default is 20
    prefix : str, optional
        String to be placed in front of the progress bar. The default is empty.
    silent : bool, optional
        If True this function doesn't return anything, which is the default. If 
        false the print statement is returned to be able to add something before 
        printing it.
        
    Returns
    -------        
    string : None or str
        the string that is outputted to the screen for the purpose of adding
        something after the string. It can be put before but one needs to take
        into account the '\\\\r' as the first character. Use string[1:] to remove it.
        If `silent` is set to True, nothing is returned. This behaviour is the default.

    Examples
    --------
        >>> for pct in range(0, 101):
                progress_bar(pct, frac = False)
                time.sleep(0.1)
        >>> |###           | 15%
            
        >>> for pct in range(0, 101):
                pct /= 100
                progress_bar(pct, line = False)
                time.sleep(0.1)
        >>> 15%
    """
    
    if frac:
        s *= 100
    s = int(s)
    width = int(width)
    
    f = 100//width
    w = 100//f

    # \r will put the cursor back at the beginning of the previous
    # print statement, if no carriage return has passed
    string = f"\r{str(prefix)}"
    
    if line:
        string += f"|{s//f * '#'}{(w - s//f) * ' '}| "
    
    string += f"{s:3}%"
    
    print(string, end="")
    
    if not silent:
        return string


def update_line(w_str):
    """
    Taking a string and overwrites (flushes) the previously printed line
    
    Parameters
    ----------
    w_str : str
        String to print
    """
    w_str = str(w_str)
    sys.stdout.write("\b" * len(w_str))
    sys.stdout.write(" " * len(w_str))
    sys.stdout.write("\b" * len(w_str))
    sys.stdout.write(w_str)
    sys.stdout.flush()


