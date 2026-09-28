'''
Dummy Tools to make Opencode Understand how to create tools for the LLM to call
'''

from typing import List 

def addition(nums : List[float]) -> float :
    '''
        Adds a list of numbers and returns their sum.
    '''
    output = 0 
    for i in range(len(nums)):
        output+=nums[i]
    return output


# Functions the LLM is allowed to call. Add new tools here.
TOOLS = [addition]
