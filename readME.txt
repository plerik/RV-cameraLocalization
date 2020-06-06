Readme of the whole project

made by Erik Plesko
May 2020 (updated 6.6.2020)
subject Robotski vid


Folders:

ArucoMarkers
    files and examples related to Aruco markers. Generation of marker images
    and detection of markers on the photo or in video.

CameraCalibration
    -> chessDetectionVideo.py
        Detection of chess in input video. Extracts and ouputs photos where chess is nicely seen.
        On frames that are not blurred the chess detections is tried, if chess is found the photo is saved.

    -> chessCalibrationFromPhotos.py
        photos achieved with chessDetectionVideo.py (or by other means) are used to calibrate the camera.
        Photos with no chess are not used. (Using both scripts we do chess detection twice, but in the sake of
        transparency and modularity. And as calibration is not a frequent process that seems reasonable.)

GeometryTransformations (not really used)
    Geometry functions which are also included in additionalFunctions.py

CalibersAndArucoCodes
     images and .pdf files of chessboard caliber and aruco markers of known dimensions
     when you need to print those.

Other_testing_code
    Examples and test-code for various operations that I needed during the working
    on the project.

PhoneVideos
    Folder meant to store videos recorded with my phone.

RVSeminarAplikacija ((( Main folder of the project)))
    This folder includes the main scripts that I need for the project (beside calibration
    and preparation of the markers).
    Naming: 
            V stands for version - those are the main versions of the project. 
            T stands for test, - things I thought I should test but are not needed
                                for the actual project
            S stands for support. - processes or routines that will be useful elsewhere
                                around the project so I put them somewhere where I 
                                can take them out and use them.

            Number after the letter tells the time-order of the code being developed.
            name after explains what the code is about.


Videos
    folder for (mostly demo) videos


