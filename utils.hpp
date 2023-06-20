#include <fstream>
#include <cassert>

class Csv {
 private:
  bool is_open = true;

 public:
  std::ofstream ofs;
  Csv(const char *file) : ofs(std::ofstream(file)) {}

  void close() {
    assert(is_open);
    is_open = false;
    ofs.close();
  }

  void open(const char *file) {
    assert(!is_open);
    is_open = true;
    ofs.open(file);
  }

  class Row {
   private:
    Csv &parent;
    bool is_first = true;

   public:
    Row(Csv &parent) : parent(parent) {}
    ~Row() { parent.ofs << std::endl; }

    Row &content(double x) {
      if (is_first) {
        parent.ofs << x;
        is_first = false;
      } else {
        parent.ofs << "," << x;
      }
      return *this;
    }
  };

  Row new_row() { return Row(*this); }

  template <typename Number>
  void save_mtx(const char *filename, int nrow, int ncol, const Number *data) {
    open(filename);
    for (int i = 0; i < nrow; i++) {
      auto row = new_row();
      for (int j = 0; j < ncol; j++) {
        row.content(data[i * ncol + j]);
      }
    }
    close();
  }
};