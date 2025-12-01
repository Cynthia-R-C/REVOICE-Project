# Basic, semi-pseudo code for destuttering
# Cynthia Chen 11/9/2025

import re

# Top-most level
def destutter(audio, txt, stutter_model):
    '''Take in: original audio packet + transcribed txt of that packet from STT
    Output: destuttered txt? Unsure how to deal with timestamps yet'''

    # 1. Classify stuttering type
    stutter_types = get_stutter_types(audio, stutter_model)  # returns a list of stutter types detected, e.g. [/p, /r]

    # 2. Remove stuttering from txt based on classified stutter types
    if '/p' in stutter_types:
        txt = remove_p_stutter(audio, txt)
    if '/b' in stutter_types:
        txt = remove_b_stutter(audio, txt)
    if '/r' in stutter_types:
        txt = remove_r_stutter(txt)
    if '/wr' in stutter_types:
        txt = remove_wr_stutter(txt)
    if '/i' in stutter_types:
        txt = remove_i_stutter(audio, txt)

    return audio, txt

def remove_r_stutter(txt):
    '''Remove repetitions of initial sounds in words from the transcribed text'''
    # Simple approach: look for patterns like s-s-s-
    # note special case - if a stuttered word is like 'b-b-bow-tie', we want to keep the full word 'bow-tie'
    # Also include multi-letter starts, e.g. "I am st-st-st-stuttering" -> "I am stuttering"
    # Works for repeating prefixes of up 1-3 letters

    pattern = r'\b([a-zA-Z]{1,3})(?:-\1){1,}-([a-zA-Z]+)'
    new = re.sub(pattern, r'\2', txt)
    # This is pretty insane, I just found out you could do this, the logic is as such:
    # \b ensures a word boundary
    # ([a-zA-Z]{1,3}) captures the first one to three letters
    # (?:-\1){1,} matches one or more repetitions of that previously identified group followed by -.
        # ?: means a non-capturing group, so we don't store this part for back-referencing
        # -\1 means “a dash followed by the same letter we captured before.” \1 refers back to the first (...) group.
        # {1, } means “one or more times.”
    # -([a-zA-Z]+) captures the rest of the real word (e.g., “ow” in “b-b-bow”).
    # The replacement \2 reconstructs the proper word (“bow”) by just keeping the second group.
    
    return new

def remove_wr_stutter(txt):
    '''Remove whole-word repetitions from the transcribed text'''
    # Simple approach: look for patterns like "I I I want" -> "I want"
    # Only removes if repetitions are consecutive and >= 3
    pattern = r'\b([a-zA-Z]+)(?: \1){2,}\b'
    new = re.sub(pattern, r'\1', txt)
    return new


# Testing
if __name__ == "__main__":
    # Simple test for remove_r_stutter
    test_txt = "I want a b-b-bow-tie a-a-and I st-st-stutter."
    print("Original:", test_txt)
    destuttered_txt = remove_r_stutter(test_txt)
    print("Destuttered:", destuttered_txt)

    # Simple test for remove_wr_stutter
    test_txt2 = "I I I want to to to go very very much."
    print("Original:", test_txt2)
    destuttered_txt2 = remove_wr_stutter(test_txt2)
    print("Destuttered:", destuttered_txt2)