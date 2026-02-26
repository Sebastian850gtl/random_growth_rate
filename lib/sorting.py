
def find_pos(T,x):
    """ T is x sorted list, a is an element that needs to be put in that list, algorithm uses a dichotomy to find
    the position of a 
    
    returns the index the item x should have"""

    n = len(T)

    a, b = -1, n

    while b-a > 1:
        c = int((a + b)/2)
        #print(a,b,c, T[c])
        if x >= T[c]:
            a = c
        else:
            b = c
    return b

if __name__ == "__main__":
    T = [1,2,4,5,9,12,13,14]
    x = 0
    print(len(T))
    print(find_pos(T,x))

    import bisect

    pos = bisect.bisect_right(T, x)
    print(pos)
