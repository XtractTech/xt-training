import sys


class Tee(object):
    """Object that replicates the functionality of the `tee` shell command."""

    def __init__(self, logfile):
        """Constructor for Tee class.
        
        Arguments:
            logfile {str} -- Path to destination file.
        """
        self.file = open(logfile, "w")
        self.stdout = sys.stdout
        sys.stdout = self

    def close(self):
        if self.stdout is not None:
            sys.stdout = self.stdout
            self.stdout = None
        if self.file is not None:
            self.file.close()
            self.file = None

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __del__(self):
        self.close()