// This program converts a set of paired images to a lmdb/leveldb by storing them
// as Datum proto buffers.
// Usage:
//   convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME
//
// where ROOTFOLDER is the root folder that holds all the images, and LISTFILE
// should be a list of files as well as their labels, in the format as
//   subfolder1/file1.JPEG 7
//   ....

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "opencv2/opencv.hpp"
#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;

DEFINE_bool(gray, false,
  "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
  "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
  "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
  "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
  "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
  "Optional: What type should we encode the image as ('png','jpg',...).");

#ifdef USE_OPENCV
void CVMatToDatum(const cv::Mat& cv_img1, const cv::Mat& cv_img2, Datum* datum) {
  CHECK(cv_img1.depth() == CV_8U) << "Image data type must be unsigned byte";
  CHECK(cv_img2.depth() == CV_8U) << "Image data type must be unsigned byte";

  datum->set_channels(cv_img1.channels() + cv_img2.channels());
  datum->set_height(cv_img1.rows);
  datum->set_width(cv_img1.cols);
  datum->clear_data();
  datum->clear_float_data();
  datum->set_encoded(false);
  int datum_channels = datum->channels();
  int datum_height = datum->height();
  int datum_width = datum->width();
  int datum_size = datum_channels * datum_height * datum_width;
  std::string buffer(datum_size, ' ');
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img1.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = 0; c < datum_channels / 2; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  for (int h = 0; h < datum_height; ++h) {
    const uchar* ptr = cv_img2.ptr<uchar>(h);
    int img_index = 0;
    for (int w = 0; w < datum_width; ++w) {
      for (int c = datum_channels / 2; c < datum_channels; ++c) {
        int datum_index = (c * datum_height + h) * datum_width + w;
        buffer[datum_index] = static_cast<char>(ptr[img_index++]);
      }
    }
  }
  datum->set_data(buffer);
}

bool ReadImageToDatum(const string& filename1, const string& filename2, const int label,
  const int height, const int width, const bool is_color,
  const std::string & encoding1, const std::string & encoding2, Datum* datum) {
  cv::Mat cv_img1 = ReadImageToCVMat(filename1, height, width, is_color);
  cv::Mat cv_img2 = ReadImageToCVMat(filename2, height, width, is_color);
  if (cv_img1.data && cv_img2.data) {
    //if (encoding.size()) {
    //  if ((cv_img.channels() == 3) == is_color && !height && !width &&
    //    matchExt(filename, encoding))
    //    return ReadFileToDatum(filename, label, datum);
    //  std::vector<uchar> buf;
    //  cv::imencode("." + encoding, cv_img, buf);
    //  datum->set_data(std::string(reinterpret_cast<char*>(&buf[0]),
    //    buf.size()));
    //  datum->set_label(label);
    //  datum->set_encoded(true);
    //  return true;
    //}


    CVMatToDatum(cv_img1, cv_img2, datum);
    datum->set_label(label);
    return true;
  }
  else {
    return false;
  }
}
#endif  // USE_OPENCV



int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
    "format used as input for Caffe.\n"
    "Usage:\n"
    "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
    "The ImageNet dataset for the training demo is at\n"
    "    http://www.image-net.org/download-images\n");
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_imageset");
    return 1;
  }

  const bool is_color = !FLAGS_gray;
  const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::pair<std::string, std::string>, int> > lines;
  std::string filename1;
  std::string filename2;
  int label;
  while (infile >> filename1 >> filename2 >> label) {
    lines.push_back(std::make_pair(std::make_pair(filename1, filename2), label));
  }
  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);

  // Create new DB
  scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
  db->Open(argv[3], db::NEW);
  scoped_ptr<db::Transaction> txn(db->NewTransaction());

  // Storing to db
  std::string root_folder(argv[1]);
  Datum datum;
  int count = 0;
  int data_size = 0;
  bool data_size_initialized = false;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    bool status;
    std::string enc1 = encode_type;
    std::string enc2 = encode_type;
    //if (encoded && !enc1.size() && !enc2.size()) {
    //  // Guess the encoding type from the file name
    //  string fn1 = lines[line_id].first.first;
    //  size_t p1 = fn1.rfind('.');
    //  if (p1 == fn1.npos)
    //    LOG(WARNING) << "Failed to guess the encoding of '" << fn1 << "'";
    //  enc1 = fn1.substr(p1);
    //  std::transform(enc1.begin(), enc1.end(), enc1.begin(), ::tolower);

    //  string fn2 = lines[line_id].first.second;
    //  size_t p2 = fn2.rfind('.');
    //  if (p2 == fn2.npos)
    //    LOG(WARNING) << "Failed to guess the encoding of '" << fn2 << "'";
    //  enc2 = fn2.substr(p1);
    //  std::transform(enc2.begin(), enc2.end(), enc2.begin(), ::tolower);
    //}
    
    status = ReadImageToDatum(root_folder + lines[line_id].first.first,
      root_folder + lines[line_id].first.second,
      lines[line_id].second, resize_height, resize_width, is_color,
      enc1, enc2, &datum);
    if (status == false) continue;
    if (check_size) {
      if (!data_size_initialized) {
        data_size = datum.channels() * datum.height() * datum.width();
        data_size_initialized = true;
      }
      else {
        const std::string& data = datum.data();
        CHECK_EQ(data.size(), data_size) << "Incorrect data field size "
          << data.size();
      }
    }
    // sequential
    string key_str = caffe::format_int(line_id, 8) + "_" + lines[line_id].first.first;

    // Put in db
    string out;
    CHECK(datum.SerializeToString(&out));
    txn->Put(key_str, out);

    if (++count % 1000 == 0) {
      // Commit db
      txn->Commit();
      txn.reset(db->NewTransaction());
      LOG(INFO) << "Processed " << count << " files.";
    }
  }
  // write the last batch
  if (count % 1000 != 0) {
    txn->Commit();
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
