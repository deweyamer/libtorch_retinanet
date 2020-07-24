//
// Created by duwei on 2020/7/22.
//

#include <iostream>
#include "torch/script.h"
#include "torch/torch.h"
#include "ATen/TensorAccessor.h"
#include "torchvision/nms.h"
#include<opencv2/opencv.hpp>
#include"opencv2/highgui/highgui.hpp"
#include <map>
#include <random>
#include <time.h>


using namespace std;

map<int,string> coco = {
        {0,"person"},{1,"bicycle"},{2,"car"},{3,"motorcycle"},{4,"airplane"},{5,"bus"},{6,"train"},{7,"truck"},{8,"boat"},{9,"traffic light"},{10,"fire hydrant"},{11,"stop sign"},{12,"parking meter"},{13,"bench"},{14,"bird"},{15,"cat"},{16,"dog"},{17,"horse"},{18,"sheep"},{19,"cow"},{20,"elephant"},{21,"bear"},{22,"zebra"},{23,"giraffe"},{24,"backpack"},{25,"umbrella"},{26,"handbag"},{27,"tie"},{28,"suitcase"},{29,"frisbee"},{30,"skis"},{31,"snowboard"},{32,"sports ball"},{33,"kite"},{34,"baseball bat"},{35,"baseball glove"},{36,"skateboard"},{37,"surfboard"},{38,"tennis racket"},{39,"bottle"},{40,"wine glass"},{41,"cup"},{42,"fork"},{43,"knife"},{44,"spoon"},{45,"bowl"},{46,"banana"},{47,"apple"},{48,"sandwich"},{49,"orange"},{50,"broccoli"},{51,"carrot"},{52,"hot dog"},{53,"pizza"},{54,"donut"},{55,"cake"},{56,"chair"},{57,"couch"},{58,"potted plant"},{59,"bed"},{60,"dining table"},{61,"toilet"},{62,"tv"},{63,"laptop"},{64,"mouse"},{65,"remote"},{66,"keyboard"},{67,"cell phone"},{68,"microwave"},{69,"oven"},{70,"toaster"},{71,"sink"},{72,"refrigerator"},{73,"book"},{74,"clock"},{75,"vase"},{76,"scissors"},{77,"teddy bear"},{78,"hair drier"},{79,"toothbrus"}
};


string toString(float x)
{
    int y = x*100;
    return "0."+to_string(y);
}


int main() {

    cv::Mat img = cv::imread("/home/duwei/000467.jpg");

    float max_wh = max(img.rows,img.cols);
    float scale = max_wh/640;
    if(scale<1)
        scale = 640/max_wh;
    //cout<<scale<<endl;
    //cout<<round(scale * img.rows)<<" "<<round(scale*img.cols);
    cv::Mat img_resize;
    int xx = round(scale * img.cols);
    int yy = round(scale * img.rows);
    if((xx%2)!=0)
        xx--;
    if((yy%2)!=0)
        yy--;
    cv::Size dsize = cv::Size(xx, yy);

    cv::resize(img,img_resize,dsize,0,0,CV_INTER_AREA);
    cv::imwrite("img.jpg",img_resize);
    int extra_pixel=0;
    int flag = 1;   //1 for cols 0 for rows
    if(img_resize.rows ==640)
    {
        extra_pixel = (640-img_resize.cols)/2;
    } else{
        extra_pixel = (640-img_resize.rows)/2;
        flag=0;
    }
    //cout<<"flag="<<flag<<endl;
    //cout<<extra_pixel_cols<<" "<<extra_pixel_rows<<endl;
    //cout<<extra_pixel_cols<<" "<<extra_pixel_cols+img_resize.cols<<endl;
    cv::Mat new_image = cv::Mat::zeros(640, 640,CV_8UC3);
    //cv::Mat new_image = cv::Mat::zeros(640, 640,CV_8UC3);


    if(flag)
    {
        for(int i=0;i<new_image.rows;i++)
        {
            for(int j=0;j<new_image.cols;j++)
            {
                if(j<=extra_pixel || j>=extra_pixel+img_resize.cols)
                {
                    new_image.at<cv::Vec3b>(i,j)[0]=0;
                    new_image.at<cv::Vec3b>(i,j)[1]=0;
                    new_image.at<cv::Vec3b>(i,j)[2]=0;
                }
                else {
                    new_image.at<cv::Vec3b>(i, j)[0] = img_resize.at<cv::Vec3b>(i, j-extra_pixel)[0];
                    new_image.at<cv::Vec3b>(i, j)[1] = img_resize.at<cv::Vec3b>(i, j-extra_pixel)[1];
                    new_image.at<cv::Vec3b>(i, j)[2] = img_resize.at<cv::Vec3b>(i, j-extra_pixel)[2];
                }

            }
        }
    }
    else
    {
        for(int i=0;i<new_image.rows;i++)
        {
            for(int j=0;j<new_image.cols;j++)
            {
                if(i<=extra_pixel || i>=extra_pixel+img_resize.rows)
                {
                    new_image.at<cv::Vec3b>(i,j)[0]=0;
                    new_image.at<cv::Vec3b>(i,j)[1]=0;
                    new_image.at<cv::Vec3b>(i,j)[2]=0;
                }
                else {
                    new_image.at<cv::Vec3b>(i, j)[0] = img_resize.at<cv::Vec3b>(i-extra_pixel, j)[0];
                    new_image.at<cv::Vec3b>(i, j)[1] = img_resize.at<cv::Vec3b>(i-extra_pixel, j)[1];
                    new_image.at<cv::Vec3b>(i, j)[2] = img_resize.at<cv::Vec3b>(i-extra_pixel, j)[2];
                }

            }
        }
    }
    cv::imwrite("/home/duwei/a.jpg",new_image);
    cv::Mat fl_new_image;
    new_image.convertTo(fl_new_image,CV_32FC3,1/255.0);
    cv::imwrite("/home/duwei/fl_a.jpg",fl_new_image);

    auto tensor_img = torch::from_blob(fl_new_image.data,{1,fl_new_image.rows,fl_new_image.cols,3}, torch::kFloat32);
    tensor_img = tensor_img.permute({0,3,1,2});
    tensor_img[0][0] = tensor_img[0][0].sub_(0.485).div_(0.229);
    tensor_img[0][1] = tensor_img[0][1].sub_(0.456).div_(0.224);
    tensor_img[0][2] = tensor_img[0][2].sub_(0.406).div_(0.225);
    //cout<<tensor_img<<endl;

    //cout<<tensor_img<<endl;
    //tensor_img.print();


    torch::jit::script::Module module;
    try {
        module = torch::jit::load("/home/duwei/retinanet/retinanet.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(tensor_img);
    //cout<<img_resize.rows<<endl;
    torch::jit::IValue output = module.forward(inputs);
    auto output1 = output.toTuple();
    auto scores = output1->elements().at(0).toTensor();
    auto boxIndex = output1->elements().at(1).toTensor();
    auto boxCoord = output1->elements().at(2).toTensor();

    //auto idx = torch::where(scores>0.5);
    auto result_score = scores.accessor<float,1>();
    auto result_boxCoord = boxCoord.accessor<float,2>();
    auto result_boxIndex = boxIndex.accessor<long,1>();

    for(int i=0;i<result_score.size(0);i++)
    {
        float score = result_score[i];
        if(score>0.5)
        {
            int x1 = result_boxCoord[i][0];
            int y1 = result_boxCoord[i][1];
            int x2 = result_boxCoord[i][2];
            int y2 = result_boxCoord[i][3];
            int label = result_boxIndex[i];
            //cout<<x1<<" "<<y1<<" "<<x2<<" "<<y2<<" "<<label<<endl;

            score = round(score*100)/100;


             random_device rd;
             default_random_engine e(rd());


            uniform_int_distribution<unsigned> u(0,255);
            auto r = u(e);
            auto g = u(e);
            auto b = u(e);
            cv::rectangle(new_image,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(r,g,b),2);
            string text = coco[label]+" "+toString(score);
            cv::putText(new_image,text,cv::Point(x1,y2-2),cv::FONT_HERSHEY_COMPLEX,0.4,cv::Scalar(255,255,255),1,cv::LINE_AA);


            if(flag==0)
            {
                y1-=extra_pixel;
                y2-=extra_pixel;
            }
            else
            {
                x1-=extra_pixel;
                x2-=extra_pixel;
            }
            x1/=scale;
            y1/=scale;
            x2/=scale;
            y2/=scale;
            cv::rectangle(img,cv::Point(x1,y1),cv::Point(x2,y2),cv::Scalar(r,g,b),2);
            cv::putText(img,text,cv::Point(x1,y2-2),cv::FONT_HERSHEY_COMPLEX,0.4,cv::Scalar(255,255,255),1,cv::LINE_AA);

        }

    }
    cv::imwrite("final.jpg",new_image);
    cv::imwrite("final2.jpg",img);
    //std::cout << "output is good" <<std::endl;


}
