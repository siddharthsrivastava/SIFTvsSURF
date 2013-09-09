#-------------------------------------------------
#
# Project created by QtCreator 2013-09-06T22:54:36
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = SIFTvsSURF
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    FalsePositives.cpp \
    CumulativeData.cpp

unix:!mac {
    message("* Using settings for Unix/Linux.")
    INCLUDEPATH += /usr/local/include/opencv

    LIBS += -L/usr/local/lib/ \
        -lopencv_core \
        -lopencv_highgui \
        -lopencv_imgproc
}

## OpenCV settings for Mac OS X
macx {
    message("* Using settings for Mac OS X.")
    INCLUDEPATH += /usr/local/include/opencv

    LIBS += -L/usr/local/lib/ \
        -lopencv_core \
        -lopencv_highgui \
        -lopencv_imgproc
}

## OpenCV settings for Windows and OpenCV 2.4.2
win32 {
    message("* Using settings for Windows.")
    INCLUDEPATH += "C:\\OpenCV\\opencv\\build\\x64\\mingw\\bin\\install\\include" \
                   "C:\\OpenCV\\opencv\\build\\x64\\mingw\\bin\\install\\include\\opencv" \
                   "C:\\OpenCV\\opencv\\build\\x64\\mingw\\bin\\install\\include\\opencv2"

    LIBS += -L"C:\\OpenCV\\opencv\\build\\x64\\mingw\\bin\\install\\lib" \
        -lopencv_core244d \
        -lopencv_highgui244d \
        -lopencv_imgproc244d \
        -lopencv_features2d244d \
        -lopencv_nonfree244d \
        -lopencv_flann244d \
        -lopencv_calib3d244d
}

HEADERS += \
    FalsePositives.h \
    CumulativeData.h

