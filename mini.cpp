#include <iostream>
#include "torch/script.h"
#include "torch/torch.h"
#include "torchvision/nms.h"
#include "torchvision/ROIAlign.h"
#include "torchvision/ROIPool.h"
#include "torchvision/empty_tensor_op.h"
#include<opencv2/opencv.hpp>
#include"opencv2/highgui/highgui.hpp"

using namespace std;

//static auto registry =
//        torch::RegisterOperators()
//                .op("torchvision::nms", &nms);


int main() {
    cv::Mat img = cv::imread("/home/duwei/000467.jpg");
    //img.convertTo(img,CV_32F);
    cv::imwrite("/home/duwei/img.jpg",img);
    cv::Mat img_resize;
    //img_resize.convertTo(img_resize,CV_32F);
    //cout<<img_resize.type()<<endl;

    cv::Size dsize = cv::Size(640, 640);
    cv::resize(img,img_resize,dsize,0,0,CV_INTER_AREA);
    cv::imwrite("/home/duwei/resize.jpg",img_resize);



    auto tensor_img = torch::from_blob(img_resize.data,{1,img_resize.rows,img_resize.cols,3}, torch::kByte);

    tensor_img = tensor_img.toType(torch::kFloat);
    tensor_img = tensor_img.div(255);
    tensor_img = tensor_img.permute({0,3,1,2});
    tensor_img[0][0] = tensor_img[0][0].sub_(0.485).div_(0.229);
    tensor_img[0][1] = tensor_img[0][1].sub_(0.456).div_(0.224);
    tensor_img[0][2] = tensor_img[0][2].sub_(0.406).div_(0.225);
    //tensor_img = tensor_img.permute({0,3,1,2});

    torch::jit::script::Module module;
    try {
        module = torch::jit::load("/home/duwei/retinanet/retinanet_cuda.pt");
    }
    catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }
    torch::DeviceType device_type; //设置Device类型
    device_type = torch::kCUDA;  //torch::kCUDA  and torch::kCPU
    torch::Device device(device_type, 0);

    //模型转到GPU中去
    module.to(device);
    module.eval();
    torch::cuda::is_available(); //判断是否支持GPU加速
    // Create a vector of inputs.
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(tensor_img.to(device));


    //指定执行位置
    // Execute the model and turn its output into a tensor.
    auto output = module.forward(inputs).toTuple();
    auto scores = output->elements()[0].toTensor();
    auto boxIndex = output->elements()[1].toTensor();
    auto boxCoord = output->elements()[2].toTensor();
    cout<<scores<<endl;
    cout<<boxIndex<<endl;
    cout<<boxCoord<<endl;
    //at::Tensor output1 = module.forward(inputs).toTuple()->elements()[1].toTensor();
    //at::Tensor output2 = module.forward(inputs).toTuple()->elements()[2].toTensor();

    //std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';

}