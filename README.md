# face-morpher

Averages faces together: label landmarks, triangulate, warp each face onto the
midway shape, blend.

## setup

```
pip install -r requirements.txt
```

Also needs `shape_predictor_68_face_landmarks.dat` (dlib's 68-point model) in
the repo root: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2

## usage

Drop photos in `assets/`, then:

```
python main.py
```

Prompts you to pick which images to merge (or enter for all). Shows each
face's landmarks/triangulation, then each face warped onto the midway shape,
then the final averaged result.

See `Notes.md` for the how/why behind the approach.