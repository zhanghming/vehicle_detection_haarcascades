#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace cv;

const int KEY_SPACE = 32;
const int KEY_ESC = 27;

CvHaarClassifierCascade *cascade;
CvMemStorage            *storage;


//set the path components
string imageDirL = "image_2/";
string imageDirR = "image_3/";
string laserDir = "velodyne/";
string calibFileName = "calib.txt";
string poseFileName = "pose.txt";
string imgPatt = "%06d.png";
string laserPatt = "%06d.bin";


void detect(IplImage *img);
void readImage(const std::string& imagePath,int imgIdx,cv::Mat& imgRead); //read image from kitti dataset

//根目录所在路径
string seqDirPath = "/home/zhanghm/Datasets/KITTI_Dataset/odometry/01/";
string calibFilePath = seqDirPath+calibFileName;


int main(int argc, char** argv)
{
  std::cout << "Using OpenCV " << CV_MAJOR_VERSION << "." << CV_MINOR_VERSION << "." << CV_SUBMINOR_VERSION << std::endl;
  
  CvCapture *capture;
  IplImage  *frame;
  int input_resize_percent = 100;
  
  if(argc < 3)
  {
    std::cout << "Usage " << argv[0] << " cascade.xml video.avi" << std::endl;
    return 0;
  }

  if(argc == 4)
  {
    input_resize_percent = atoi(argv[3]);
    std::cout << "Resizing to: " << input_resize_percent << "%" << std::endl;
  }

  cascade = (CvHaarClassifierCascade*) cvLoad(argv[1], 0, 0, 0);
  storage = cvCreateMemStorage(0);
  capture = cvCaptureFromAVI(argv[2]);

  assert(cascade && storage && capture);

  cvNamedWindow("video", CV_WINDOW_NORMAL);

  IplImage* frame1 = cvQueryFrame(capture);
  frame = cvCreateImage(cvSize((int)((frame1->width*input_resize_percent)/100) , (int)((frame1->height*input_resize_percent)/100)), frame1->depth, frame1->nChannels);

  int key = 0;
  int start = 0; //sequence number
  do
  {
    //1)读入图片数据
    string imgFileName = seqDirPath+imageDirL;
    Mat imgSrc;
    readImage(imgFileName,start,imgSrc);

    if(imgSrc.empty())
      break;

    //cvResize(frame1, frame);
    IplImage frame_detect = imgSrc;

    detect(&frame_detect);
    ++start;

    key = cvWaitKey(33);

    if(key == KEY_SPACE)
      key = cvWaitKey(0);

    if(key == KEY_ESC)
      break;

  }while(1);

  cvDestroyAllWindows();
  cvReleaseImage(&frame);
  cvReleaseCapture(&capture);
  cvReleaseHaarClassifierCascade(&cascade);
  cvReleaseMemStorage(&storage);

  return 0;
}

void detect(IplImage *img)
{
  CvSize img_size = cvGetSize(img);
  CvSeq *object = cvHaarDetectObjects(
    img,
    cascade,
    storage,
    1.1, //1.1,//1.5, //-------------------SCALE FACTOR
    1, //2        //------------------MIN NEIGHBOURS
    0, //CV_HAAR_DO_CANNY_PRUNING
    cvSize(0,0),//cvSize( 30,30), // ------MINSIZE
    img_size //cvSize(70,70)//cvSize(640,480)  //---------MAXSIZE
    );

  std::cout << "Total: " << object->total << " cars detected." << std::endl;
  for(int i = 0 ; i < ( object ? object->total : 0 ) ; i++)
  {
    CvRect *r = (CvRect*)cvGetSeqElem(object, i);
    cvRectangle(img,
      cvPoint(r->x, r->y),
      cvPoint(r->x + r->width, r->y + r->height),
      CV_RGB(255, 0, 0), 2, 8, 0);
  }

  cvShowImage("video", img);
}


void readImage(const std::string& imagePath,int imgIdx,cv::Mat& imgRead)
{
  char temp[100];
  sprintf(temp,"%06d.png",imgIdx);
  string full_image_path = imagePath+ static_cast<string>(temp);
//    string full_image_path = imagePath+ "000000.png";

  imgRead = cv::imread(full_image_path);

}

