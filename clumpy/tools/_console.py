import sys

from ._path import path_split, create_directories

def title_heading(n):
    if n > 0:
        return('\n'+''.join(['#' for i in range(n)])+' ')
    else:
        return('')

class Transcript(object):

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.logfile = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.logfile.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

def start_log(path):
    """
    Start transcript, appending print output to given filename.

    Parameters
    ----------
    path : str
        Log file path.
    """
    
    folder_path, file_name, file_ext = path_split(path)
    create_directories(folder_path)
    
    sys.stdout = Transcript(path)

def stop_log():
    """
    Stop transcript and return print functionality to normal
    """
    sys.stdout.logfile.close()
    sys.stdout = sys.stdout.terminal
