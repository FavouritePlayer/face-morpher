# face-morpher
Ok so use CS 180 to figure out how to do this
General understanding of the steps:

1. Label the face with symmetrical points
Spec says to label by hand, maybe just auto label and then wait for a validation step?

Now, you need to provide a triangulation of these points that will be used for morphing. You can compute a triangulation any way you like, or even define it by hand. A Delaunay triangulation (see delaunay and related functions) is a good choice since it does not produce overly skinny triangles. You can compute the Delaunay triangulation on either of the point sets (but not both -- the triangulation has to be the same throughout the morph!). But the best approach would probably be to compute the triangulation at a midway shape (i.e. mean of the two point sets) to lessen the potential triangle deformations.


2. "Compute the midway shape"
This would involve: 
    1) computing the average shape (a.k.a the average of each keypoint location in the two faces), 
    2) warping both faces into that shape, and 
    3) averaging the colors together. The main task in warping the faces into the average shape is implementing an affine warp for each triangle in the triangulation from the original images into this new shape. This will involve computing an affine transformation matrix A between two triangles:

    A = computeAffine(tri1_pts,tri2_pts)

Ok so figure out what exactly this means? Define "affine warp"? Probably use of a "affine transformation matrix"


