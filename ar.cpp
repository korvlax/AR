#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d.hpp>

#include <iostream>
#include <fstream>
#include <cstring>
#include <string>
#include <cmath>

using namespace std;
using namespace cv;

const Size chess_dim = Size(6,9);
const float square_length = 0.017;
const float side_length = 0.1; //cm
const int frames_per_second = 30;
const float square_similarity =  0.7;


/* NOTES
 For n ≥ 4 and the points are coplanar and there is no triplets of collinear points, the solution is
unique. Use CV_P3P parameter in solePnP function 
*/

void create_world_pts(Size boardSize, float square_length, vector<Point3f>& corners)
{
    for(int i = 0; i < boardSize.height ; i++)
    {
        for(int j = 0; j < boardSize.width; j++)
        {
            corners.push_back(Point3f(i*square_length,j*square_length,0.0f));
        }
    }
}

void create_square_pts(float side_length, vector<Point3f>& square_world_pts)
{
    square_world_pts.push_back(Point3f(side_length, 0.0, 0.0));
    square_world_pts.push_back(Point3f(side_length, side_length, 0.0));
    square_world_pts.push_back(Point3f(0.0, side_length, 0.0));
    square_world_pts.push_back(Point3f(0.0, 0.0, 0.0));
}

void get_chess_corners(vector<Mat> images, vector<vector<Point2f> >& all_found_corners, bool show_results = false)
{
    cout << images.size() << endl;
    for(vector<Mat>::iterator iter = images.begin(); iter != images.end(); iter++)
    {
        vector<Point2f> point_buff;
        bool found = findChessboardCorners(*iter, chess_dim, point_buff, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        cout << found << endl;
        if(found)
        {
            all_found_corners.push_back(point_buff);

        }
        if(show_results)
        {
            drawChessboardCorners(*iter, chess_dim, point_buff, found);
            imshow("Looking for corners : ", *iter);
            waitKey(0);
        }
    }
}

void cam_calibration(vector<Mat> calibration_imgs,Size chess_dim, float square_length, Mat& cam_matrix, Mat& dist_coeff)
{

    vector<vector<Point2f> > chess_board_img_space_pts;
    get_chess_corners(calibration_imgs, chess_board_img_space_pts, false);

    vector<vector<Point3f> > world_space_corner_pts(1);

    //world pos coords are the same for all images. Therefore we will copy this matrix.
    create_world_pts(chess_dim, square_length, world_space_corner_pts[0]);
    world_space_corner_pts.resize(chess_board_img_space_pts.size(), world_space_corner_pts[0]);
    vector<Mat> rVecs, tVecs;
    dist_coeff = Mat::zeros(8,1,CV_32FC1);

    cout << chess_board_img_space_pts.size()<< endl;

    calibrateCamera(world_space_corner_pts, chess_board_img_space_pts, chess_dim, cam_matrix, dist_coeff, rVecs, tVecs);
    cout << rVecs[0] << endl;
    
}

bool save_cam_calibration(char* name, Mat cam_matrix, Mat dist_coeff)
{
    ofstream outStream;
    outStream.open(name);
    if(outStream)
    {

        for(int r = 0 ; r < cam_matrix.rows ; r++)
        {
            for(int c = 0 ; c < cam_matrix.cols ; c++)
            {
                double value = cam_matrix.at<double>(r,c);
                outStream << value << endl;
            }
        }
        for(int r = 0 ; r < dist_coeff.rows ; r++)
        {
            for(int c = 0 ; c < dist_coeff.cols ; c++)
            {
                double value = dist_coeff.at<double>(r,c);
                outStream << value << endl;
            }
        }
    outStream.close();
    return true;
    }
    return false;
}

bool load_cam_calibration(Mat& cam_matrix, Mat& dist_coeff, string path)
{
    ifstream file;
    file.open("Camera_and_distcoeff");
    vector<float> values;
    string line;
    int index = 0;
    cout << cam_matrix << endl;
    while(std::getline(file, line))
    {
       values.push_back(stod(line));        
    }
    for(int i  = 0 ; i < 3 ; i++)
    {
        for(int j = 0 ; j < 3 ; j++)
        {
            float value = values[index++];
            cam_matrix.at<float>(i,j) = value;
        }
    }
    //cam_matrix.at<double>()
    cout << "cam matrix: " << endl;
    cout << cam_matrix << endl;
    file.close();
}

void start_calibration_mode(VideoCapture cap, Mat frame, Mat draw_to_frame, Mat& cam_matrix, Mat& dist_coeff )
{
    vector<Mat> saved_imgs;
    vector<vector<Point2f> > marked_corners, rejected_candidates;        

    while(true)
    {
        if(!cap.read(frame))
            break;

        vector<Vec2f> found_pts;
        bool found = false;
        
        found = findChessboardCorners(frame, chess_dim, found_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
        frame.copyTo(draw_to_frame);
        drawChessboardCorners(draw_to_frame, chess_dim, found_pts, found);
        if(found)
            imshow("Webcam",draw_to_frame);
        else
            imshow("Webcam", frame);

        char character = waitKey(1000 / frames_per_second);

        //We will calibrate camera in live mode. User moves aroudn checkerboard and decides if it wants to use 
        // the image as data for the calibration

        if(character == ' ')
        {
            if(found)
            {
                cout << "Saving image" << endl;
                Mat temp;
                frame.copyTo(temp);
                saved_imgs.push_back(temp);
            }else
            {
                cout << "need to detect corners to save!" << endl;
            }

        }
        switch(character)
        {
        case 13:
            //start calibration when enter is pressed
            //minimum amout of images is 1
            if(saved_imgs.size() > 1)
            {
                cam_calibration(saved_imgs, chess_dim, square_length, cam_matrix, dist_coeff);
                save_cam_calibration("Camera_and_distcoeff",cam_matrix,dist_coeff);
                return;
            }else
            {
                cout << "Need " << 1 - saved_imgs.size() << " more images" << endl;
            }
            break;

        case 27:
            //exit
            return;
            break;
        }
    }

}

void create_cube(vector<Point3f>& cube_pts, float side, float location)
{
    for(int i = 0 ; i < 2 ; i++)
    {
        for(int j = 0 ; j < 2 ; j++)
        {
            for(int k = 0; k < 2 ; k++)
            {
                cube_pts.push_back(Point3f(location + i*side, location + j*side, location + k*side));
            }
        }
    }

}

void find_cam_pose_chessboard(Mat& img, Mat& cam_matrix, Mat& dist_coeff, Mat& rvec, Mat& tvec, vector<Point3f>& world_pts,vector<Point2f>& img_pts)
{
    bool found = findChessboardCorners(img, chess_dim, img_pts, CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE);
    if(!found)
    {
        cout << "cant find corners" << endl;
        return;
    }else
    {
        cout << "found corners" << endl;
    }
    create_world_pts(chess_dim, square_length, world_pts);
    solvePnP(world_pts, img_pts, cam_matrix, dist_coeff, rvec, tvec);
}

bool find_squares(vector<vector<Point> >& big_contours,
 vector<vector<Point> >& contours_img,
  vector<vector<Point> >& square_contours,
  vector<vector<Point> >& input_verts,
  vector<Point2f>& square_corners)
{
    double max_area = 0;
    int max_contour_index = -1;

    for(int i = 0 ; i < big_contours.size() ; i++)
    {
        float epsilon = 0.1*arcLength(big_contours[i], true);
        approxPolyDP(big_contours[i], input_verts[i], epsilon, true);
        if (input_verts[i].size() == 4)
        {
            double sim = matchShapes(big_contours[i], contours_img[0],1, 0.0);
            if(sim < square_similarity)
            {
                cout << "Square detected wiith coords: " << endl;
                for(int j = 0 ; j < 4 ; j++) cout << input_verts[i][j] << endl;

                double square_area = contourArea(big_contours[i]);
                if(square_area > max_area)
                {
                    square_area = max_area;
                    max_contour_index = i;
                } 

                cout << "Area = " << square_area << endl;
            }
        }
    }
    if(max_contour_index != -1)
    {
        square_contours.push_back(big_contours[max_contour_index]);
        for(int j = 0 ; j < 4 ; j++)
        {
            square_corners.push_back(input_verts[max_contour_index][j]);
        }
        return true;
    }else
    {
        return false;
    }
}

void draw_squares(Mat& contour_drawing, Mat& frame, vector<vector<Point> >& square_contours, vector<Vec4i>& hierarchy)
{
    for( int i = 0; i< square_contours.size(); i++ )
        {
            Scalar color = Scalar(0,0,255);
            drawContours( contour_drawing, square_contours, i, color, 2, 8, hierarchy, 0, Point() );
            drawContours( frame, square_contours, i, color, 2, 8, hierarchy, 0, Point() );
        }
}

void find_pose(vector<Point2f>& square_corners, Mat& cam_matrix, Mat& dist_coeff, Mat& rvec, Mat& tvec)
{
    vector<Point3f> square_world_pts;
    create_square_pts(side_length, square_world_pts);
    solvePnP(square_world_pts, square_corners, cam_matrix, dist_coeff, rvec, tvec   );
    cout << rvec << endl;
    cout << tvec << endl;
}

void draw_coords_and_cube(Mat rvec, Mat tvec, Mat cam_matrix,  Mat dist_coeff, Mat& frame)
{
    double a = side_length;
                    
    vector<Point2f> pts_in_img1;
    vector<Point3f> pts_in_world1;
    Point3f p1(0.0, 0.0, 0.0);
    
    Point3f p2(a, 0.0, 0.0);
    pts_in_world1.push_back(p1);
    pts_in_world1.push_back(p2);
    //red -> x-axis
    projectPoints(pts_in_world1,rvec,tvec,cam_matrix,dist_coeff, pts_in_img1);
    arrowedLine(frame, pts_in_img1[0], pts_in_img1[1], Scalar(0,0,255),2,8,0,0.1);

    vector<Point2f> pts_in_img2;
    vector<Point3f> pts_in_world2;
    Point3f p3(0.0, a, 0.0);
    pts_in_world2.push_back(p1);
    pts_in_world2.push_back(p3);
    // y -> green
    projectPoints(pts_in_world2,rvec,tvec,cam_matrix,dist_coeff, pts_in_img2);
    arrowedLine(frame, pts_in_img2[0], pts_in_img2[1], Scalar(0,255,0),2,8,0,0.1);

    vector<Point2f> pts_in_img3;
    vector<Point3f> pts_in_world3;
    Point3f p4(0.0, 0.0, -a);
    pts_in_world3.push_back(p1);
    pts_in_world3.push_back(p4);
    // z - axis -> blue
    projectPoints(pts_in_world3,rvec,tvec,cam_matrix,dist_coeff, pts_in_img3);
    arrowedLine(frame, pts_in_img3[0], pts_in_img3[1], Scalar(255,0,0),2,8,0,0.1);

    vector<Point2f> cube_pts_img1;
    vector<Point2f> cube_pts_img2;
    vector<Point3f> cube_pts_world1;
    vector<Point3f> cube_pts_world2;

    cube_pts_world1.push_back(Point3f(0 ,0,0));
    cube_pts_world1.push_back(Point3f(a,0,0));
    cube_pts_world1.push_back(Point3f(a,a,0));
    cube_pts_world1.push_back(Point3f(0,a,0));

    cube_pts_world2.push_back(Point3f(0,0,-a));
    cube_pts_world2.push_back(Point3f(a,0,-a));
    cube_pts_world2.push_back(Point3f(a,a,-a));
    cube_pts_world2.push_back(Point3f(0,a,-a));

    projectPoints(cube_pts_world1, rvec, tvec, cam_matrix, dist_coeff, cube_pts_img1);
    projectPoints(cube_pts_world2, rvec, tvec, cam_matrix, dist_coeff, cube_pts_img2);

    //Draws line which will together represent a cube.
    for(int i = 0 ; i < 4; i++)
    {
        line(frame, cube_pts_img1[i], cube_pts_img1[(i+1) % 4], Scalar(100,50,100));
    }
    for(int i = 0 ; i < 4; i++)
    {
        line(frame, cube_pts_img2[i], cube_pts_img2[(i+1) % 4], Scalar(100,50,100));
    }
        for(int i = 0 ; i < 4; i++)
    {
        line(frame, cube_pts_img1[i], cube_pts_img2[i], Scalar(100,50,100));
    }
}

int main(int argc, char** argv) {
    Mat frame;
    Mat draw_to_frame;
    Mat square_img = imread("/home/korvlax/Documents/Code/CV/ArProject/src/square.png");

    Mat cam_matrix = Mat::eye(3,3, CV_32FC1);
    Mat dist_coeff;

    cout << "OpenCV version : " << CV_VERSION << endl;

    VideoCapture cap(0);
    if(!cap.isOpened())
    {
        cap.open(0);
        cout << "cant open cam" << endl;
        return -1;
    }

    namedWindow("drawover",WINDOW_AUTOSIZE);
    namedWindow("contour", WINDOW_AUTOSIZE);

    system("../src/DisableAutoFocus.sh");

    cout << "Do you want to recalibrate cam? Enter yes if yes, no if no..." << endl;
    string recalibrate;
    cin >> recalibrate;

    if(recalibrate == "yes")
    {   
        start_calibration_mode(cap, frame, draw_to_frame, cam_matrix, dist_coeff);
    }else
    {
        load_cam_calibration(cam_matrix, dist_coeff, "Camera_and_distcoeff");

        //code that tries to find camera from four corners of square

        float move_x = 0.0;
        
        while(true)
        {
            if(!cap.read(frame))
                break;
            
            Mat frame_gray, square_img_gray;
            int thresh = 30;
            int max_thresh = 255;
            RNG rng(12345);

            cvtColor( square_img, square_img_gray, COLOR_BGR2GRAY );
            blur( square_img_gray, square_img_gray, Size(3,3) );
            
            cvtColor( frame, frame_gray, COLOR_BGR2GRAY );
            blur( frame_gray, frame_gray, Size(3,3) );

            Mat canny_output, canny_output_img;
            vector<vector<Point> > contours, contours_img;
            vector<Vec4i> hierarchy, hierarchy_img;

            Canny( frame_gray, canny_output, thresh, thresh*2, 3 );
            dilate(canny_output, canny_output, Mat(), Point(-1,-1));
            findContours(canny_output, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0,0));

            //filter out small squares from contours
            double max_area = 0;
            double thresh_area = 4000.0;
            vector<vector<Point> > big_contours;
            for(int i = 0 ; i < contours.size() ; i++)
            {
                double area = contourArea(contours[i]);
                if(area > thresh_area)
                {
                    big_contours.push_back(contours[i]);
                }
                max_area = area;
            }

            Canny( square_img_gray, canny_output_img, thresh, thresh*2, 3 );
            dilate(canny_output_img, canny_output_img, Mat(), Point(-1,-1));
            findContours(canny_output_img, contours_img, hierarchy_img,RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0,0));
            vector<vector<Point> > img_verts(contours_img.size()), input_verts(big_contours.size());

            float epsilon_img = 0.1*arcLength(contours_img[0], true);
            approxPolyDP(contours_img[0], img_verts[0], epsilon_img, true);

            vector<vector<Point> > square_contours;
            Mat contour_drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
            vector<Point2f> square_corners;
            

            if(big_contours.size() > 0)
            {
                bool found = find_squares(big_contours, contours_img, square_contours, input_verts, square_corners);
                if(found)
                {
                    draw_squares(contour_drawing, frame, square_contours, hierarchy);
                    
                    Mat rvec, tvec;
                    find_pose(square_corners, cam_matrix, dist_coeff, rvec, tvec);

                    //Draws coordinate system and the cube.
                    draw_coords_and_cube(rvec, tvec, cam_matrix, dist_coeff, frame);
                } 
            }
            imshow("drawover", frame);
            imshow("contour", contour_drawing);

            char character = waitKey(1000 / frames_per_second);
        }        
    }
    
    return 0;
}
