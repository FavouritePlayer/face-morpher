# face-morpher
Deepmind here I come
Ok so use CS 180 to figure out how to do this
Additonal:
https://learnopencv.com/face-morph-using-opencv-cpp-python/

General understanding of the steps:

1. Label the face with symmetrical points
Spec says to label by hand, maybe just auto label and then wait for a validation step?

Now, you need to provide a triangulation of these points that will be used for morphing. You can compute a triangulation any way you like, or even define it by hand. A Delaunay triangulation (see delaunay and related functions) is a good choice since it does not produce overly skinny triangles. You can compute the Delaunay triangulation on either of the point sets (but not both -- the triangulation has to be the same throughout the morph!). But the best approach would probably be to compute the triangulation at a midway shape (i.e. mean of the two point sets) to lessen the potential triangle deformations.

Use http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
to get face labeler.

NVM switching to MediaPipe for forehead coverage!!!

Why use triangles? Cant just move pixles. Since a face is not rigid, we have to divide it into traingles and morph each triangle face
How to decide how traingles are drawn given the 68 points?
"Compute Delaunay triangulation" so that triangles with a small enough angle are not created. Since triangles have to be uniform run over one face only.
How do we represent triangles as a datastructure?

Answer first compute average of each point to form a midpoint labeling
Compute Delaunay triangulation on that to get the specified triangles
Apply the same triangle pattern to both original faces
Calculate the "affine transformation matrix" 


2. "Compute the midway shape"
This would involve: 
    1) computing the average shape (a.k.a the average of each keypoint location in the two faces), 
    2) warping both faces into that shape, and 
    3) averaging the colors together. The main task in warping the faces into the average shape is implementing an affine warp for each triangle in the triangulation from the original images into this new shape. This will involve computing an affine transformation matrix A between two triangles:

    A = computeAffine(tri1_pts,tri2_pts)

Ok so figure out what exactly this means? Define "affine warp"? Probably use of a "affine transformation matrix"

3. Normalizing coords across photos
Cant average raw pixel points from two different photos, (500,300) means something different in a 4000x3000 pic vs a 1000x800 one
So normalize_coords divides each point by that image's own width/height to get [0,1] coords, now theyre comparable and can be averaged for the midway shape

But photos also have different aspect ratios (portrait vs landscape), if you resize width/height independently to match youll stretch the face
So crop_images pipeline runs first and center-crops every image to the same aspect ratio (square for now) before any labeling/normalizing happens, so everything lines up

Ok so at affine transformation step
First warp onto overlapping canvas then average colors. COlors part is easy.
Warp part?


