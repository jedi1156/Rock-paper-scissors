For 2 people on one camera
OpenCV.pass_semester(semester_id: 5)

# setup on osx

OpenCV version 2.4.6.1 has broken wabcam support on osx is broken. Use 2.4.6:

Change `/usr/local/Library/Taps/homebrew-science/opencv.rb` :

```
-  url 'http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.6.1/opencv-2.4.6.1.tar.gz'
-  sha1 'e015bd67218844b38daf3cea8aab505b592a66c0'
+  url 'http://downloads.sourceforge.net/project/opencvlibrary/opencv-unix/2.4.6/opencv-2.4.6.tar.gz'
```
