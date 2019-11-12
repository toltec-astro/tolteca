import re

SLICE_TEMPLATES = [
    ('s', r'(?P<i>[+-]?\d+)'),
    ('sp', r'\((?P<i>[+-]?\d+)\)'),
    ('a', r'::?'),
    ('ri-', r'(?P<i>[+-]?\d+)::?'),
    ('ri-k', r'(?P<i>[+-]?\d+)::(?P<k>[+-]?\d+)'),
    ('r-j', r':(?P<j>[+-]?\d+):?'),
    ('r-jk', r':(?P<j>[+-]?\d+):(?P<k>[+-]?\d+)'),
    ('rij', r'(?P<i>[+-]?\d+):(?P<j>[+-]?\d+):?'),
    ('rijk', r'(?P<i>[+-]?\d+):(?P<j>[+-]?\d+):(?P<k>[+-]?\d+)'),
    ('r--k', r'::(?P<k>[+-]?\d+)'),
    ('l', r'\.\.\.'),
    ('eb', r'\[(?P<e>[+-]?\d+(,[+-]?\d+)*,?)\]'),
    ('ep', r'\((?P<e>[+-]?\d+(,[+-]?\d+)+,?)\)'),
    ('ep1', r'\((?P<e>[+-]?\d+,)\)'),
]
SLICE_TEMPLATES = [(k, re.compile(v)) for k, v in SLICE_TEMPLATES]


def tokenize_slice_groups(string):
    # tokenize groups
    groups = []
    sbuf = []
    expecting = {'(': ')', '[': ']'}
    pbbuf = []
    LEGAL_CHARS = '0123456789()[]+-:.'
    WHITESPACE_CHARS = ' \t'

    for c in string:
        if c in WHITESPACE_CHARS:
            pass
        elif c == ',':
            if len(pbbuf) not in (0, 2):
                sbuf.append(c)
            else:
                groups.append(''.join(sbuf))
                sbuf.clear()
                pbbuf.clear()
        elif c in LEGAL_CHARS:
            sbuf.append(c)
            if c in '([':
                if pbbuf:
                    raise ValueError('too many brackets in axis {}'.format(
                        len(groups)))
                pbbuf.append(c)
            elif c in ')]':
                if not pbbuf:
                    raise ValueError('brackets not match in axis {}'.format(
                        len(groups)))
                if c != expecting[pbbuf[0]]:
                    raise ValueError('brackets not match in axis {}'.format(
                        len(groups)))
                pbbuf.append(c)
        else:
            raise ValueError('illegal char `{}\''.format(c))
    groups.append(''.join(sbuf))
    return groups


def parse_slice_group(string):
    for name, tem in SLICE_TEMPLATES:
        matched = tem.fullmatch(string)
        if matched:
            if name[0] == 's':
                return int(matched.group('i'))
            if name[0] == 'a':
                return slice(None, None, None)
            if name[0] == 'r':
                i, j, k = None, None, None
                if 'i' in name:
                    i = int(matched.group('i'))
                if 'j' in name:
                    j = int(matched.group('j'))
                if 'k' in name:
                    k = int(matched.group('k'))
                return slice(i, j, k)
            if name[0] == 'l':
                return ...
            # if name[0] == 'e'
            return list(map(int, filter(None, matched.group('e').split(','))))
    raise ValueError('illegal group "{}"'.format(string))


def parse_slice(string):
    groups = tokenize_slice_groups(string)
    if groups == ['']:
        raise ValueError('index must not be empty')
    if groups and groups[-1] == '':
        del groups[-1]
    index = tuple(map(parse_slice_group, groups))
    if index.count(...) > 1:
        raise ValueError('ellipsis may occur at most once')
    return index
