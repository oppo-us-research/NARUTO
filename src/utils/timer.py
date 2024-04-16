"""
MIT License

Copyright (c) 2024 OPPO

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


import numpy as np
from time import time


class Timer():
    """Timer class to count time and do time analysis
    """

    def __init__(self, items=None):
        """
        Args:
            items (list/str): list of items to be counted, each item is a str
        """
        self.timers = {}
        if items is not None:
            self.add(items)

    def add(self, item, group=None):
        """add item to the timer
        
        Args:
            item (str/list): item name
            group (str): group name of the item
        """
        if isinstance(item, list):
            for i in item:
                self.timers[i] = {
                    'name': i,
                    'time': 0,
                    'is_counting': False,
                    'duration': [],
                    'group': group
                }
        elif isinstance(item, str):
            self.timers[item] = {
                    'name': item,
                    'time': 0,
                    'is_counting': False,
                    'duration': [],
                    'group': group
                }
        else:
            assert False, "only list or str is accepted."
    
    def start(self, item, group=None):
        """Start timer for an item

        Args:
            item (str): timer name
            group (str): group name for the item
        """
        if self.timers.get(item, -1) == -1:
            self.add(item, group)

        assert not(self.timers[item]['is_counting']),  "Timer for {} has started already.".format(item)
        
        self.timers[item]['is_counting'] = True
        self.timers[item]['time'] = time()
    
    def end(self, item):
        """Stop timer for an item

        Args:
            item (str): timer name
        """
        assert self.timers[item]['is_counting'], "Timer for {} has not started.".format(item)
        
        duration = time() - self.timers[item]['time']
        self.timers[item]['duration'].append(duration)
        self.timers[item]['is_counting'] = False

    def get_last_timing(self, item: str) -> float:
        """ get last timing for the item
    
        Args:
            item: timer name
    
        Returns:
            time: Unit: seconds
        """
        return self.timers[item]['duration'][-1]
    
    def time_analysis(self, method="median"):
        """Time analysis of the items
        """
        print("----- time breakdown -----")
        # group items according to groups
        group_timers = {'single': []}
        for key in sorted(self.timers.keys()):
            group_name = self.timers[key]['group']
            if group_name is not None:
                if group_timers.get(group_name, -1) == -1:
                    group_timers[group_name] = []
                group_timers[group_name].append(self.timers[key])
            else:
                group_timers['single'].append(self.timers[key])
        
        # display times
        for group_name, members in group_timers.items():
            print("Group [{}]: ".format(group_name))
            group_avg_times = []
            for member in members:
                if method == "mean":
                    avg_time = np.asarray(member['duration']).mean()
                elif method == "median":
                    avg_time = np.median(np.asarray(member['duration']))
                else:
                    raise NotImplementedError
                group_avg_times.append(avg_time)
                print("\t[{}]: {:.03f}ms".format(member['name'], avg_time*1000))