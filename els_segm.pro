#-------------------------------------------------
#
# Project created by QtCreator 2014-01-06T09:36:41
#
#-------------------------------------------------

QT       += core

QT       -= gui

TARGET = els_segm
CONFIG   += console
CONFIG   -= app_bundle

TEMPLATE = app


SOURCES += main.cpp \
    lbp.cpp \
    histogram.cpp \
    chanvese.cpp

unix: CONFIG += link_pkgconfig
unix: PKGCONFIG += opencv

HEADERS += \
    LBP.hpp \
    lbp.hpp \
    histogram.hpp \
    chanvese.h
