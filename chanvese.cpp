#include "chanvese.h"
#include <iostream>

using namespace std;

chanvese::chanvese()
{
}

void heaviside2(Mat z, Mat &H, float epsilon, Mat mask){
    int cols = z.cols;
    int rows = z.rows;
    H = Mat::zeros(rows,cols,CV_32F);
    for(int i = 0; i<rows; i++){
        for(int j = 0; j<cols; j++){
            if(mask.at<uint8_t>(i,j) & 1){
            H.at<float>(i,j) = (1.0/2.0)*(1+(2/M_PI)*atan(z.at<float>(i,j)/epsilon));
            }
        }
    }
}

void initphi0(Mat &phi0,int rows, int cols, int radius, int shift,Mat mask){
    Mat phi = Mat::zeros(rows,cols,CV_8U);
    Mat phimask = Mat::zeros(rows,cols,CV_8U);
    phi0 = Mat::zeros(rows,cols,CV_32F);
    int maxcirch = floor(cols/(radius*2+10));
    int maxcircv = floor(rows/(radius*2+10));
    for(int i = 0; i<maxcirch-1; i++){
        for(int j = 0; j<maxcircv-1; j++){
            int x = (radius*2+10)*i+radius+shift;
            int y = (radius*2+10)*j+radius+shift;
            circle(phi,Point(x,y),radius,Scalar(1),-1);
        }
    }
    phi.copyTo(phimask,mask);
    distanceTransform(phimask,phi0,CV_DIST_L1,CV_DIST_MASK_3);
    /*float x;
    float y;
    for (int i = 0; i < rows; ++i){
      for (int j = 0; j < cols; ++j){
          if(mask.at<uint8_t>(i,j) & 1){
        x = float(i) - rows/2.0 - shift;
        y = float(j) - cols/2.0 - shift;
        phi0.at<float>(i,j) = 900.0/(900.0 + x*x + y*y) - 0.5;
        //phi0.at<float>(i,j) = sqrt(pow(x,2)+pow(y,2));
          }
      }
    }*/
}

void computeC1C2(Mat im_in, Mat phi0,float &c1, float &c2, Mat mask){
    Mat H, Hneg;
    Mat im_in_mask;
    im_in.copyTo(im_in_mask,mask);
    heaviside2(phi0,H,0.01,mask);
    Hneg = 1 - H;
    Mat mult1 = Mat::zeros(phi0.rows, phi0.cols,CV_32F);
    //Mat mult1a = Mat::zeros(phi0.rows, phi0.cols,CV_32F);
    Mat mult2 = Mat::zeros(phi0.rows, phi0.cols,CV_32F);
    //Mat mult2a = Mat::zeros(phi0.rows, phi0.cols,CV_32F);
    cv::multiply(im_in_mask,H,mult1);
    cv::multiply(im_in_mask,Hneg,mult2);
    //cv::multiply(mult1,mask,mult1a);
    //cv::multiply(mult2,mask,mult2a);
    c1 = cv::sum(mult1).val[0]/cv::sum(H).val[0];
    c2 = cv::sum(mult2).val[0]/cv::sum(Hneg).val[0];
}

void calcDeltah(Mat phi, Mat &deltah, float h){
    deltah = Mat::zeros(phi.rows,phi.cols,CV_32F);
    for (int i = 0; i < phi.rows; ++i){
        for (int j = 0; j < phi.cols; ++j){
            deltah.at<float>(i,j) = h/(M_PI*(pow(h,2) + pow(phi.at<float>(i,j),2)));
        }
    }
}

void calcL(Mat phi, Mat deltah, float &L){
    Mat grad_x, grad_y, grad, Lmat;
    Sobel( phi, grad_x, CV_32F, 1, 0);
    Sobel( phi, grad_y, CV_32F, 0, 1);
    addWeighted( grad_x, 0.5, grad_y, 0.5, 0, grad );
    grad = cv::abs(grad);
    multiply(grad,deltah,Lmat);
    L = sum(Lmat).val[0]/(Lmat.cols*Lmat.rows);
}

void updatePhi(Mat im_in, Mat phi, Mat &phinew, Mat mask, float h, float deltat, float mu, float p, float nu, float lambda1, float lambda2, float c1, float c2){
    phinew = Mat::zeros(phi.rows,phi.cols,CV_32F);
    float C1, C2, C3, C4, term1, F, F1, F2, F3, F4, L, Pij;
    float eps = 1e-16;
    Mat deltah;
    calcDeltah(phi, deltah, h);
    calcL(phi,deltah, L);
    for (int i = 1; i < phi.rows-1; ++i){
        for (int j = 1; j < phi.cols-1; ++j){
            if(mask.at<uint8_t>(i,j) & 1){
            C1 = 1/sqrt(eps + pow((phi.at<float>(i+1,j) - phi.at<float>(i,j)),2)
                                   + pow((phi.at<float>(i,j+1) - phi.at<float>(i,j-1)),2)/4.0);
            C2 = 1/sqrt(eps + pow((phi.at<float>(i,j) - phi.at<float>(i-1,j)),2)
                                   + pow((phi.at<float>(i-1,j+1) - phi.at<float>(i-1,j-1)),2)/4.0);
            C3 = 1/sqrt(eps + pow((phi.at<float>(i+1,j) - phi.at<float>(i-1,j)),2)/4.0
                                   + pow((phi.at<float>(i,j+1) - phi.at<float>(i,j)),2));
            C4 = 1/sqrt(eps + pow((phi.at<float>(i+1,j-1) - phi.at<float>(i-1,j-1)),2)/4.0
                                   + pow((phi.at<float>(i,j) - phi.at<float>(i,j-1)),2));
            term1 = deltat*deltah.at<float>(i,j)*mu*p*pow(L,p-1);
            F = h/(h+term1*(C1+C2+C3+C4));
            term1 = term1/(h+term1*(C1+C2+C3+C4));

            F1 = term1*C1;
            F2 = term1*C2;
            F3 = term1*C3;
            F4 = term1*C4;

            Pij = phi.at<float>(i,j) - deltat*deltah.at<float>(i,j)*(nu + lambda1*pow(im_in.at<float>(i,j)-c1,2) - lambda2*pow(im_in.at<float>(i,j)-c2,2));
            phinew.at<float>(i,j) = F1*phi.at<float>(i+1,j)+ F2*phi.at<float>(i-1,j) + F3*phi.at<float>(i,j+1)+ F4*phi.at<float>(i,j-1) + F*Pij;
            }
        }
    }
    for (unsigned int i = 0; i < im_in.rows; ++i){
      phinew.at<float>(i,0) = phinew.at<float>(i,1);
      phinew.at<float>(i,im_in.cols-1) = phinew.at<float>(i,im_in.cols-2);
    }
    for (unsigned int j = 0; j < im_in.cols; ++j){
      phinew.at<float>(0,j) = phinew.at<float>(1,j);
      phinew.at<float>(im_in.rows-1,j) = phinew.at<float>(im_in.rows-2,j);
    }
}

void reinitPhi(Mat &phi, int numIter, double h, double deltat){
    double a, b, c, d, x, G, Q;
    bool fStop = false;
    unsigned int M;
    Mat psiOld, psiOut;
    psiOut = phi.clone();
    for (unsigned int k = 0; k < numIter && fStop == false; ++k)
    {
      psiOld = psiOut.clone();
      for (int i = 1; i < phi.rows-1; ++i){
          for (int j = 1; j < phi.cols-1; ++j){
              a = (phi.at<float>(i,j) - phi.at<float>(i-1,j))/h;
              b = (phi.at<float>(i+1,j) - phi.at<float>(i,j))/h;
              c = (phi.at<float>(i,j) - phi.at<float>(i,j-1))/h;
              d = (phi.at<float>(i,j+1) - phi.at<float>(i,j))/h;

          if (phi.at<float>(i,j) > 0)
            G = sqrt(std::max(std::max(a,0.0)*std::max(a,0.0),std::min(b,0.0)*std::min(b,0.0))
                   + std::max(std::max(c,0.0)*std::max(c,0.0),std::min(d,0.0)*std::min(d,0.0))) - 1.0;
          else if (phi.at<float>(i,j) < 0)
            G = sqrt(std::max(std::min(a,0.0)*std::min(a,0.0),std::max(b,0.0)*std::max(b,0.0))
                   + std::max(std::min(c,0.0)*std::min(c,0.0),std::max(d,0.0)*std::max(d,0.0))) - 1.0;
          else
            G = 0;

          x = (phi.at<float>(i,j) >= 0)?(1.0):(-1.0);
          psiOut.at<float>(i,j) = psiOut.at<float>(i,j) - deltat*x*G;
        }
      }
      Q = 0.0;
      M = 0.0;
      for (int i = 0; i < phi.rows; ++i){
        for (int j = 0; j < phi.cols; ++j){
            if (abs(psiOld.at<float>(i,j)) <= h){
                ++M;
                Q += abs(psiOld.at<float>(i,j) - psiOut.at<float>(i,j));
            }
        }
      }
      if (M != 0)
        Q = Q/((double)M);
      else
        Q = 0.0;
      if (Q < deltat*h*h){
        fStop = true;
      }
    }
    phi = psiOut.clone();
}

void findZero(Mat im_in, Mat &edges, int fg, int bg){
    edges = Mat(im_in.rows, im_in.cols, CV_8U, Scalar(bg));
    for (int i = 0; i < im_in.rows; ++i){
        for (int j = 0; j < im_in.cols; ++j){
            if (i > 0 && i < (im_in.rows-1)  && j > 0 && j < (im_in.cols-1)){
                if (0 == im_in.at<float>(i,j)){
                    if (0 != im_in.at<float>(i-1,j-1)
                            || 0 != im_in.at<float>(i-1,j)
                            || 0 != im_in.at<float>(i-1,j+1)
                            || 0 != im_in.at<float>(i,j-1)
                            || 0 != im_in.at<float>(i,j+1)
                            || 0 != im_in.at<float>(i+1,j-1)
                            || 0 != im_in.at<float>(i+1,j)
                            || 0 != im_in.at<float>(i+1,j+1)){
                        edges.at<uint8_t>(i,j) = fg;
                    }
                }else{
                    if (abs(im_in.at<float>(i,j)) < abs(im_in.at<float>(i-1,j-1))
                            && (im_in.at<float>(i,j)>0) != (im_in.at<float>(i-1,j-1)>0))
                        edges.at<uint8_t>(i,j) = fg;
                    else if (abs(im_in.at<float>(i,j)) < abs(im_in.at<float>(i-1,j))
                             && (im_in.at<float>(i,j)>0) != (im_in.at<float>(i-1,j)>0))
                        edges.at<uint8_t>(i,j) = fg;
                    else if (abs(im_in.at<float>(i,j)) < abs(im_in.at<float>(i-1,j+1))
                             && (im_in.at<float>(i,j)>0) != (im_in.at<float>(i-1,j+1)>0))
                        edges.at<uint8_t>(i,j) = fg;
                    else if (abs(im_in.at<float>(i,j)) < abs(im_in.at<float>(i,j-1))
                             && (im_in.at<float>(i,j)>0) != (im_in.at<float>(i,j-1)>0))
                        edges.at<uint8_t>(i,j) = fg;
                    else if (abs(im_in.at<float>(i,j)) < abs(im_in.at<float>(i,j+1))
                             && (im_in.at<float>(i,j)>0) != (im_in.at<float>(i,j+1)>0))
                        edges.at<uint8_t>(i,j) = fg;
                    else if (abs(im_in.at<float>(i,j)) < abs(im_in.at<float>(i+1,j-1))
                             && (im_in.at<float>(i,j)>0) != (im_in.at<float>(i+1,j-1)>0))
                        edges.at<uint8_t>(i,j) = fg;
                    else if (abs(im_in.at<float>(i,j)) < abs(im_in.at<float>(i+1,j))
                             && (im_in.at<float>(i,j)>0) != (im_in.at<float>(i+1,j)>0))
                        edges.at<uint8_t>(i,j) = fg;
                    else if (abs(im_in.at<float>(i,j)) < abs(im_in.at<float>(i+1,j+1))
                             && (im_in.at<float>(i,j)>0) != (im_in.at<float>(i+1,j+1)>0))
                        edges.at<uint8_t>(i,j) = fg;
                }
            }
            if(im_in.at<float>(i,j)<0){
                //edges.at<uint8_t>(i,j) = fg;
            }
        }
    }
}

void chanvese::segment(Mat im_inrgb, Mat &im_out, Mat mask){
    Mat im_in;
    blur(im_inrgb,im_inrgb,Size(5,5));
    //cvtColor(im_inrgb,im_in,CV_RGB2GRAY);
    im_in = im_inrgb.clone();
    Mat L;
    im_in.convertTo(L,CV_32F,double(255.0/255.0));
    Mat phi0;
    initphi0(phi0,im_in.rows,im_in.cols, 30, 10, mask);
    float c1, c2;
    Mat phi, tempEdges, tempSeg;

    int numIter = 200;
    float h = 10;
    float deltat = 1;
    float mu = 1;
    float p = 1;
    float nu = 1;
    float lambda1 = 1;
    float lambda2 = 1;

    erode(mask,mask,Mat::ones(5,5,CV_8U));
    namedWindow("Process",0);
    for(int k = 0;k<numIter; k++){
        computeC1C2(L,phi0,c1,c2,mask);
        updatePhi(L,phi0,phi,mask,h,deltat,mu,p,nu,lambda1,lambda2,c1,c2);
        reinitPhi(phi,100,h,deltat);
        phi0 = phi.clone();
        findZero(phi0,tempEdges,0,255);
        //Mat binMask = tempEdges & 255;
        //im_inrgb.copyTo(tempSeg,binMask);
        //normalize(phi0,tempSeg, 0, 255, NORM_MINMAX, CV_8UC1);
        imshow("Process",tempEdges);
        waitKey(20);
    }
    Mat edges;
    findZero(phi0,edges,0,255);
    im_out = edges.clone();
}

void computeC1C2C3C4(Mat u, Mat phi1, Mat phi2, float &c1, float &c2, float &c3, float &c4, Mat mask){
    Mat H1, H1neg, H2, H2neg;
    Mat u_mask;
    u.copyTo(u_mask,mask);
    heaviside2(phi1,H1,1,mask);
    heaviside2(phi2,H2,1,mask);
    H1neg = 1 - H1;
    H2neg = 1 - H2;
    float num1, num2, num3, num4, den1, den2, den3, den4 = 0;
    for(int i = 0; i<phi1.rows; i++){
        for(int j = 0; j<phi1.cols; j++){
            num1 += u_mask.at<float>(i,j)*H1.at<float>(i,j)*H2.at<float>(i,j);
            den1 += H1.at<float>(i,j)*H2.at<float>(i,j);
            num2 += u_mask.at<float>(i,j)*H1.at<float>(i,j)*H2neg.at<float>(i,j);
            den2 += H1.at<float>(i,j)*H2neg.at<float>(i,j);
            num3 += u_mask.at<float>(i,j)*H1neg.at<float>(i,j)*H2.at<float>(i,j);
            den3 += H1neg.at<float>(i,j)*H2.at<float>(i,j);
            num4 += u_mask.at<float>(i,j)*H1neg.at<float>(i,j)*H2.at<float>(i,j);
            den4 += H1neg.at<float>(i,j)*H2.at<float>(i,j);
        }
    }
    c1 = num1/den1;
    c2 = num2/den2;
    c3 = num3/den3;
    c4 = num4/den4;
}

void update2Phi(Mat L, Mat phi1old, Mat phi2old, Mat &phi1new, Mat &phi2new, Mat mask, float h, float deltat, float nu, float c1, float c2, float c3, float c4){
    phi1new = Mat::zeros(phi1old.rows,phi1old.cols,CV_32F);
    phi2new = Mat::zeros(phi2old.rows,phi2old.cols,CV_32F);
    float C1, C2, C3, C4, C, D1, D2, D3, D4, D, m1, m2;
    Mat H1, H2;
    Mat u_mask;
    //L.copyTo(u_mask,mask);
    heaviside2(phi1old,H1,1,mask);
    heaviside2(phi2old,H2,1,mask);
    float eps = 1e-6;
    Mat deltah1, deltah2;
    calcDeltah(phi1old, deltah1, h);
    calcDeltah(phi2old, deltah2, h);
    for (int i = 1; i < phi1old.rows-1; ++i){
        for (int j = 1; j < phi1old.cols-1; ++j){
            C1 = 1/sqrt(eps + pow((phi1old.at<float>(i+1,j)-phi1old.at<float>(i,j))/h,2)+pow((phi1old.at<float>(i,j+1)-phi1old.at<float>(i,j-1))/(2*h),2));
            C2 = 1/sqrt(eps + pow((phi1old.at<float>(i,j)-phi1old.at<float>(i-1,j))/h,2)+pow((phi1old.at<float>(i-1,j+1)-phi1old.at<float>(i-1,j-1))/(2*h),2));
            C3 = 1/sqrt(eps + pow((phi1old.at<float>(i+1,j)-phi1old.at<float>(i-1,j))/(2*h),2)+pow((phi1old.at<float>(i,j+1)-phi1old.at<float>(i,j))/h,2));
            C4 = 1/sqrt(eps + pow((phi1old.at<float>(i+1,j-1)-phi1old.at<float>(i-1,j-1))/(2*h),2)+pow((phi1old.at<float>(i,j)-phi1old.at<float>(i,j-1))/h,2));
            m1 = (deltat/pow(h,2))*deltah1.at<float>(i,j)*nu;
            C = 1 + m1*(C1+C2+C3+C4);
            phi1new.at<float>(i,j) =
                    (1/C)*(phi1old.at<float>(i,j)
                           +m1*(C1*phi1old.at<float>(i+1,j)+
                                C2*phi1old.at<float>(i-1,j)+
                                C3*phi1old.at<float>(i,j+1)+
                                C4*phi1old.at<float>(i,j-1))+
                           deltat*deltah1.at<float>(i,j)*(-pow(L.at<float>(i,j)-c1,2)*H2.at<float>(i,j)
                                                          -pow(L.at<float>(i,j)-c2,2)*(1-H2.at<float>(i,j))
                                                          +pow(L.at<float>(i,j)-c3,2)*H2.at<float>(i,j)
                                                          +pow(L.at<float>(i,j)-c4,2)*(1-H2.at<float>(i,j))));
            D1 = 1/sqrt(eps + pow((phi2old.at<float>(i+1,j)-phi2old.at<float>(i,j))/h,2)+pow((phi2old.at<float>(i,j+1)-phi2old.at<float>(i,j-1))/(2*h),2));
            D2 = 1/sqrt(eps + pow((phi2old.at<float>(i,j)-phi2old.at<float>(i-1,j))/h,2)+pow((phi2old.at<float>(i-1,j+1)-phi2old.at<float>(i-1,j-1))/(2*h),2));
            D3 = 1/sqrt(eps + pow((phi2old.at<float>(i+1,j)-phi2old.at<float>(i-1,j))/(2*h),2)+pow((phi2old.at<float>(i,j+1)-phi2old.at<float>(i,j))/h,2));
            D4 = 1/sqrt(eps + pow((phi2old.at<float>(i+1,j-1)-phi2old.at<float>(i-1,j-1))/(2*h),2)+pow((phi2old.at<float>(i,j)-phi2old.at<float>(i,j-1))/h,2));
            m2 = (deltat/pow(h,2))*deltah2.at<float>(i,j)*nu;
            D = 1 + m2*(D1+D2+D3+D4);
            phi2new.at<float>(i,j) =
                    (1/D)*(phi2old.at<float>(i,j)
                           +m2*(D1*phi2old.at<float>(i+1,j)+
                                D2*phi2old.at<float>(i-1,j)+
                                D3*phi2old.at<float>(i,j+1)+
                                D4*phi2old.at<float>(i,j-1))+
                           deltat*deltah2.at<float>(i,j)*(-pow(L.at<float>(i,j)-c1,2)*H1.at<float>(i,j)
                                                          -pow(L.at<float>(i,j)-c2,2)*(1-H1.at<float>(i,j))
                                                          +pow(L.at<float>(i,j)-c3,2)*H1.at<float>(i,j)
                                                          +pow(L.at<float>(i,j)-c4,2)*(1-H1.at<float>(i,j))));
        }
    }
    for (unsigned int i = 0; i < L.rows; ++i){
      phi1new.at<float>(i,0) = phi1new.at<float>(i,1);
      phi1new.at<float>(i,L.cols-1) = phi1new.at<float>(i,L.cols-2);
      phi2new.at<float>(i,0) = phi2new.at<float>(i,1);
      phi2new.at<float>(i,L.cols-1) = phi2new.at<float>(i,L.cols-2);
    }
    for (unsigned int j = 0; j < L.cols; ++j){
      phi1new.at<float>(0,j) = phi1new.at<float>(1,j);
      phi1new.at<float>(L.rows-1,j) = phi1new.at<float>(L.rows-2,j);
      phi2new.at<float>(0,j) = phi2new.at<float>(1,j);
      phi2new.at<float>(L.rows-1,j) = phi2new.at<float>(L.rows-2,j);
    }
}

void findSeg2Phi(Mat phi1, Mat phi2, Mat &tempSeg){
    tempSeg = Mat::zeros(phi1.rows,phi1.cols,CV_32F);
    for(int i = 0; i< tempSeg.rows; i++){
        for(int j = 0; j< tempSeg.cols; j++){
            if(phi1.at<float>(i,j)<0 && phi2.at<float>(i,j)<0){
                tempSeg.at<float>(i,j) = 1;
            }
            else if(phi1.at<float>(i,j)<0 && phi2.at<float>(i,j)>0){
                tempSeg.at<float>(i,j) = 2;
            }
            else if(phi1.at<float>(i,j)>0 && phi2.at<float>(i,j)<0){
                tempSeg.at<float>(i,j) = 3;
            }
            else if(phi1.at<float>(i,j)>0 && phi2.at<float>(i,j)>0){
                tempSeg.at<float>(i,j) = 4;
            }
        }
    }
    normalize(tempSeg,tempSeg,0,255,CV_MINMAX,CV_8U);
}

void chanvese::segment2phi(Mat im_inrgb, Mat &im_out, Mat mask){
    Mat im_in;
    //blur(im_inrgb,im_inrgb,Size(5,5));
    cvtColor(im_inrgb,im_in,CV_BGR2GRAY);
    Mat L;
    im_in.convertTo(L,CV_32F,double(255.0/255.0));
    Mat phi1, phi2;
    initphi0(phi1,im_in.rows,im_in.cols, 30, 10,mask);
    initphi0(phi2,im_in.rows,im_in.cols, 30, 15,mask);
    float c1, c2, c3, c4 = 0;
    Mat phi1new, phi2new, tempEdges1, tempEdges2, tempSeg;

    int numIter = 200;
    float h = 1;
    float deltat = 0.1;
    float nu = 0.001*pow(255,2);

    //erode(mask,mask,Mat::ones(5,5,CV_8U));
    namedWindow("Phi 1",0);
    namedWindow("Phi 2",0);
    findZero(phi1,tempEdges1,0,255);
    findZero(phi2,tempEdges2,0,255);
    imshow("Phi 1",tempEdges1);
    imshow("Phi 2",tempEdges2);
    findSeg2Phi(phi1, phi2, tempSeg);
    namedWindow("Process",0);
    imshow("Process",tempSeg);
    waitKey();

    for(int n = 0; n<numIter; n++){
        computeC1C2C3C4(L, phi1, phi2, c1, c2, c3, c4, mask);
        update2Phi(L,phi1, phi2 ,phi1new, phi2new, mask, h, deltat, nu, c1, c2 , c3, c4);
        findSeg2Phi(phi1new, phi2new, tempSeg);
        phi1 = phi1new.clone();
        phi2 = phi2new.clone();
        reinitPhi(phi1,1,h,deltat);
        reinitPhi(phi2,1,h,deltat);
        findZero(phi1,tempEdges1,0,255);
        imshow("Process",tempEdges1);
        waitKey(1);
    }
    Mat edges;
    im_out = tempSeg.clone();

}
