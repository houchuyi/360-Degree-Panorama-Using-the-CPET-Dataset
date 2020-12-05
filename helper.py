def obtain_index(i):

    # if the current image is taken by the bottom row cam (i is odd)
    if i%2 != 0:
        # the top image is one of the candidate tp = i-1
        # then we take 4 more images from the left and right cameras
        # tpr = i-3 btr = i-2 tpl = i-9 btl = i-8 (cannot use + b/c index might be out of bound)
        return [i-1,i-3,i-2,i-9,i-8]

    #else i is even, the image is taken by the top row cam
    else:
        # the bottom image is one of the candidate bt = i+1
        # then we take 4 more images from the left and right cameras
        # tpr = i-2 btr = i-1 tpl = i-8 btl = i-7
        return [i+1,i-2,i-1,i-8,i-7]

def obtain_good_matches(bf,cur_index, neighb_index,dess):

    # use brute force match to find the kth nearest neighbour match
    matches = bf.knnMatch(dess[cur_index],dess[neighb_index],k=2)

    # Store all the good matches as per Lowe's ratio test
    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    return good
