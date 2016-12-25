#!/usr/bin/env python3
# vim: set fileencoding=utf-8 :

"""
Various hacks for getting around limitations / bugs in GATE, QunToken, etc.
Ideally, this file should be empty.
"""

import re

# Maximum length of a paragraph (longer P's break QunToken)
__max_p_length = 60000
__eos = re.compile(ur'[a-záéíóöőúüű]{4,}[.!?]+')

def split_for_qt(text, sep='\n\n'):
    """
    This function splits the output into chunks small enough for
    QunToken to be able to process along sentence boundaries (best effort).
    Any chunk longer than this threshold (such as those stupid "high-lit"
    books where the author thinks it is a good idea to write a chapter /
    the whole book as one very long sentence) is discarded.
    """
    if len(text) > __max_p_length:
        chunks, last_pos = [], 0
        while text:
            for m in __eos.finditer(text):
                if m.end() > __max_p_length:
                    if last_pos != 0:
                        chunks.append(text[:last_pos] + sep)
                        text = text[last_pos:]
                    else:
                        text = text[m.end():]
                    break
                else:
                    last_pos = m.end()
            else:
                chunks.append(text + sep)
                text = None
        return chunks
    else:
        return [text + sep]
