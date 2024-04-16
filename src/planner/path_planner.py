class PathPlanner():
    def __init__(self, *argv, **kwargs):
        """
    
        Args:
            *args   : Variable length argument list for positional arguments.
            **kwargs: Arbitrary keyword arguments.
    
        Returns:
            
        """
        return
    
    def start_new_plan(self):
        """ initialize a new planning request 
        """
        raise NotImplementedError

    def run_full(self):
        """ Run planner to approximately cover the whole map
        """
        raise NotImplementedError
    
    def run(self):
        """ run planner
        """
        raise NotImplementedError
    
    def find_path(self):
        """ find path
        """
        raise NotImplementedError
    
    def get_reachable_mask(self):
        """get reachable/traversability mask
        """
        raise NotImplementedError
