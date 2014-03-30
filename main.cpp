#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/ml/ml.hpp>
#include <iostream>
#include "lbp.hpp"
#include "histogram.hpp"
#include "chanvese.h"

using namespace cv;
using namespace std;

int main()
{    
    // Load and resize image
    Mat im_in = imread("images/34.JPG");
    resize(im_in,im_in,Size(800,600));
    namedWindow("Input image",0);
    imshow("Input image",im_in);
    waitKey();
    const int num_obs = im_in.cols*im_in.rows;

    // ---------------Pores Segmentation---------------------
    // LBP for texture
    Mat im_gray, im_lbp;
    int radius = 1;
    int neighbors = 4;
    cvtColor(im_in, im_gray, CV_BGR2GRAY);
    GaussianBlur(im_gray, im_gray, Size(5,5),5, 3, BORDER_CONSTANT);
    lbp::ELBP(im_gray, im_lbp, radius, neighbors);
    normalize(im_lbp, im_lbp, 0, 255, NORM_MINMAX, CV_8UC1);
    namedWindow("LBP",0);
    imshow("LBP",im_lbp);
    waitKey();



    // Gabor filter
    Mat im_gabor(im_gray.size(),CV_32F);
    Mat im_gabor_int(im_gray.size(),CV_8U);
    vector<Mat> im_gabor_set;
    Mat im_gray_float;
    Mat kernel(42,42,CV_32F);
    namedWindow("Gabor",0);
    for(int i = 0; i<8;i++){
        kernel = getGaborKernel(Size(15,15),1,(CV_PI*double (i))/4.0,6,1);
        im_gray.convertTo(im_gray_float,CV_32F,double(1.0/255),0);
        filter2D(im_gray_float,im_gabor,CV_32F,kernel);
        normalize(im_gabor, im_gabor_int, 0, 255, NORM_MINMAX, CV_8UC1);
        im_gabor_set.push_back(im_gabor.reshape(1,num_obs));
        imshow("Gabor",im_gabor_int);
        waitKey();
    }

    // Color information
    vector<Mat> imgRGB;
    split(im_in,imgRGB);
    Mat img3xN(num_obs,3,CV_8U);
    for(int i=0;i!=3;++i){
        imgRGB[i].reshape(1,num_obs).copyTo(img3xN.col(i));
    }
    img3xN.convertTo(img3xN,CV_32F,1.0/255.0);

    // Lowpass Filter
    Mat im_filtered;
    GaussianBlur(im_gray,im_filtered,Size(25,25),1);
    Mat featurefilt;
    im_filtered.reshape(1,num_obs).convertTo(featurefilt,CV_32F,1.0/255.0);


    // Feature vector
    Mat featuremat(num_obs,13,CV_32F,Scalar(0));
    Mat featurelbp = Mat(1, num_obs,CV_32F,Scalar(0));
    im_lbp.reshape(1,num_obs).convertTo(featurelbp,CV_32F,1.0/255.0);

    for(int i=0; i<num_obs;i++){
        featuremat.at<float>(i,0) = featurelbp.at<float>(i,0);
        for(int j=1; j<4;j++){
            featuremat.at<float>(i,j) = img3xN.at<float>(i,j-1);
        }
        for(int j=4; j<12;j++){
            featuremat.at<float>(i,j) = im_gabor_set[j-4].at<float>(i,0);
        }
        featuremat.at<float>(i,12) = featurefilt.at<float>(i,0);
    }
    //Mat feature_float;
    //featuremat.convertTo(feature_float,CV_32F,1.0/255.0);
    //cout << featuremat<<endl;

    // k-means
    /*Mat bestlabels, bestlabels_im;
    Mat classes;
    int k = 5;
    kmeans(feature_float,k,bestlabels,TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0),3, KMEANS_PP_CENTERS);
    bestlabels = bestlabels.reshape(1,im_in.rows);
    bestlabels.convertTo(bestlabels,CV_8U);
    normalize(bestlabels, bestlabels_im, 0, 255, NORM_MINMAX, CV_8UC1);
    namedWindow("Kmeans",0);
    imshow("Kmeans",bestlabels_im);
    waitKey();
    for(int m=0; m<k;m++){
        classes=Mat::zeros(im_in.rows,im_in.cols, CV_8U);
        for(int i=0; i<im_in.rows;i++){
            for(int j=0; j<im_in.cols;j++){
                if(bestlabels.at<u_int8_t>(i,j) == m){
                    classes.at<u_int8_t>(i,j) = 255;
                }
            }
        }
        imshow("Kmeans",classes);
        waitKey();
    }*/

    // Naive Bayes - Training
    Mat im_labels_in = imread("images/34man.bmp",0);
    resize(im_labels_in,im_labels_in,Size(800,600),0,0,INTER_NEAREST);
    namedWindow("Labels image",0);
    imshow("Labels image",im_labels_in);
    waitKey();
    Mat labelsmat(num_obs,1,CV_8U,Scalar(0));
    im_labels_in.reshape(1,num_obs).convertTo(labelsmat,CV_8U,1.0/255.0);
    Mat mask = Mat::ones(num_obs,1,CV_8U);
    for(int i= 0; i<num_obs;i++){
        if(img3xN.at<float>(i,0)<0.05 && img3xN.at<float>(i,1)<0.05 && img3xN.at<float>(i,2)<0.05){
            labelsmat.at<u_int8_t>(i,0) = 2;
            mask.at<uint>(i,0) = 0;
        }
    }
    Mat labels_float;
    labelsmat.convertTo(labels_float,CV_32F);
    Mat labels_img;
    labels_img = labelsmat.reshape(1,600).clone();
    normalize(labels_img, labels_img, 0, 255, NORM_MINMAX, CV_8UC1);
    namedWindow("Labels",0);
    imshow("Labels",labels_img);
    waitKey();

    //Mat im_ac = imread("images/35.png");
    Mat im_ac_in = im_gray(Rect(200,200,400,300)).clone();
    Mat mask_ac;
    //erode(mask,mask_ac,Mat::ones(45,45,CV_8U));
    mask_ac = Mat::ones(im_ac_in.rows,im_ac_in.cols,CV_8U);
    chanvese a;
    Mat act_cont;
    a.segment(im_ac_in,act_cont,mask_ac);

    CvNormalBayesClassifier bayes;
    bayes.train(featuremat,labels_float);

    Mat predicted;
    bayes.predict(featuremat,&predicted);
    Mat classified;
    classified = predicted.reshape(1,600);
    normalize(classified,classified,0,255,NORM_MINMAX);
    classified.convertTo(classified,CV_8U);
    namedWindow("Predicted",0);
    imshow("Predicted",classified);
    waitKey();

    /*CvSVMParams params;
    params.svm_type    = CvSVM::C_SVC;
    params.kernel_type = CvSVM::POLY;
    params.term_crit   = cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
    params.degree = 5;

    CvSVM SVM;
    SVM.train(feature_float, labels_float, Mat(), Mat(), params);
    Mat predicted(num_obs,1,CV_32F,Scalar(0));
    for(int i = 0; i<num_obs; i++){
        Mat sampleMat = feature_float.row(i);
        predicted.at<float>(i,0) = SVM.predict(sampleMat);
    }
    Mat classified;
    classified = predicted.reshape(1,600);
    normalize(classified,classified,0,255,NORM_MINMAX);
    classified.convertTo(classified,CV_8U);
    namedWindow("Predicted",0);
    imshow("Predicted",classified);
    waitKey();*/

    Mat poros = predicted.reshape(1,600) < 1 ;
    normalize(poros,poros,0,255,NORM_MINMAX);
    poros.convertTo(poros,CV_8U);
    namedWindow("Poros",0);
    imshow("Poros",poros);
    waitKey();

    return 0;
}
