from __future__ import print_function
import time
import operator
import search

if __name__ == "__main__":
    start = time.clock()

    ns = 6
    max_superposition = 5
    targets = [[ [x, y] for x in xrange(ns) for y in xrange(x + 1, ns) ]]
    for i in xrange(3, max_superposition + 1):
        targets.append([prev + [x] for prev in targets[i - 3]
                                   for x in xrange(max(prev) + 1, ns) ])
    targets = reduce(operator.__add__, targets)

    out_files = [ open("motional_{}".format(i), "w")
                  for i in xrange(2, max_superposition + 1) ]
    log_files = [ open("log_{}".format(i), "w")
                  for i in xrange(2, max_superposition + 1) ]
    completed_this_length = [ 0 ] * (max_superposition - 1)

    for target in targets:
        l = len(target)
        out_file = out_files[l - 2]
        log_file = log_files[l - 2]
        if completed_this_length[l - 2] != 0:
            print("\n", file = out_file)
            print(file = log_file)
        print("Target:", ", ".join(map(str, target)), file = out_file)
        print("Target:", ", ".join(map(str, target)), file = log_file)
        colours_printed = 0
        for (colours, angle_set) in search.search(target, log = log_file):
            if colours_printed != 0:
                print(file = out_file)
            print("- Colours:", ", ".join(colours), file = out_file)
            for angles in angle_set:
                print("-- Angles:", ", ".join(map(str, angles)),
                      file = out_file)
            colours_printed += 1
        completed_this_length[l - 2] += 1

    print("Time elapsed:", time.clock() - start)
