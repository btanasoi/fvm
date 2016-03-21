/*
 * Original utility. This one is simpler because it reads in all the data
 * from the xyz file then writes the hdf5 and xdmf files.  This is the
 * fastest approach, but fails when converting extremely large files.
 *
 * Kept as a reference.
 */
#include<iostream>
#include<fstream>

#ifndef H5_NO_NAMESPACE
#ifndef H5_NO_STD
    using std::cout;
    using std::endl;
#endif  // H5_NO_STD
#endif

#include "H5Cpp.h"
#include <stdlib.h>
#include <string.h>

#ifndef H5_NO_NAMESPACE
    using namespace H5;
#endif

using namespace std;

const char *_xstart = "<?xml version=\"1.0\" ?>\n\
<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n\
<Xdmf xmlns:xi=\"http://www.w3.org/2003/XInclude\" Version=\"2.1\">\n\
  <Domain>\n\
    <Grid Name=\"%s\" GridType=\"Collection\" CollectionType=\"Temporal\">\n";
const char *end ="    </Grid>\n\
   </Domain>\n\
</Xdmf>\n";
const char *_grid="      <Grid Name=\"%d\">\n\
        <Time Value=\"%d\"/>\n\
        <Topology TopologyType=\"Polyvertex\" Dimensions=\"%d\"/>\n\
        <Geometry GeometryType=\"XYZ\">\n\
          <DataItem NumberType=\"Float\" Format=\"HDF\" Precision=\"4\" Dimensions=\"%d 3\">%s:/%d/data</DataItem>\n\
        </Geometry>\n\
        <Attribute Name=\"element\" AttributeType=\"Scalar\" Center=\"Node\">\n\
          <DataItem NumberType=\"Int\" Format=\"HDF\" Precision=\"1\" Dimensions=\"%d\">%s:/%d/element</DataItem>\n\
        </Attribute>\n\
      </Grid>\n";

char *xstart(char *gname)
{
  static char buf[1024];
  sprintf(buf,_xstart,gname);
  return buf;
}

char *grid(int step, int num, const char *h5file)
{
  static char buf[1024];
  sprintf(buf,_grid,step,step,num,num,h5file,step,num,h5file,step);
  return buf;
}


void write_h5_int8(void *data, int nx, Group &file, char *dname)
{
  hsize_t dims[1] = {nx};
  DataSpace dspace (1, dims);
  IntType dt (PredType::NATIVE_UCHAR);
  dt.setOrder(H5T_ORDER_LE);
  DataSet ds = file.createDataSet(dname, dt, dspace);
  ds.write(data, PredType::STD_I8LE);
}

void write_h5_float32(void *data, int nx, Group &file, char *dname)
{
  hsize_t dims[2] = {nx, 3};
  DataSpace dspace (2, dims);
  DSetCreatPropList plist;
  plist.setChunk( 2, dims );
  plist.setDeflate( 6 );
  IntType dt (PredType::NATIVE_FLOAT);
  dt.setOrder(H5T_ORDER_LE);
  DataSet ds = file.createDataSet(dname, dt, dspace, plist);
  ds.write(data, PredType::IEEE_F32LE);
}

const char element_table[][3] = {
  "", /* 0 */
  "H","He",
  "Li","Be","B", "C", "N","O","F", "Ne",
  "Na","Mg","Al","Si","P","S","Cl","Ar",
  "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn","Ga","Ge","As","Se","Br","Kr",
  "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd","In","Sn","Sb","Te","I","Xe",
  "Cs","Ba",
  "La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy","Ho","Er","Tm","Yb",
  "Lu","Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg","Tl","Pb","Bi","Po","At","Rn",
  "Fr","Ra",
};

char element_number(string symbol)
{
  /* linear search. fixme */
  const int size = sizeof(element_table)/sizeof(element_table[0]);
  for (int i = 1; i < size; i++)
    if (symbol == string(element_table[i]))
      return i;
  return 0;
}

void usage(char *name) {
  cerr << "XYZ to XDMF file converter.\n";
  cerr << "usage: " << name << " [-v] filename.xyz\n";
  exit(-1);
}

int verbose_flag = 0;

int main(int argc, char *argv[])
{
  string element, comment;
  char buf[256], *elements, *element_ptr;
  float x,y,z, *data, *data_ptr;
  int num, step, count, option_char, wrote_start = 0;

  while ((option_char = getopt(argc, argv, "v")) != EOF) {
    switch (option_char)
      {
      case 'v': verbose_flag = 1; break;
      case '?': usage(argv[0]);
      }
  }

  if (optind >= argc)
    usage(argv[0]);

  string iname = argv[optind];
  string bname (iname.begin(), iname.end()-4);
  string xmfname  = bname + ".xmf";
  string h5name  = bname + ".h5";

  ofstream xout(xmfname.c_str());

  ifstream stream1(iname.c_str());
  if(!stream1)
    {
      cout << "While opening a file an error is encountered" << endl;
      return(-1);
    }

  H5File file (h5name, H5F_ACC_TRUNC);
  step = 0;

  while (!stream1.eof()) {
    // read number of atoms
    stream1 >> num;

    if (stream1.eof())
      break;

    // read comment
    stream1.getline(buf,sizeof(buf));
    stream1.getline(buf,sizeof(buf));
    if (wrote_start == 0) {
      wrote_start = 1;
      if (strlen(buf) < 2)
	strcat(buf,"Data");
      xout << xstart(buf);
    }

    data = data_ptr = (float *)malloc(num * 3 * sizeof(*data));
    elements = element_ptr = (char *)malloc(num * sizeof(*elements));
    if (data == NULL || elements == NULL) {
      cerr << "Out of memory\n";
      return(-1);
    }

    count = num;
    while(count-- && !stream1.eof())
      {
	stream1 >> element >> x >> y >> z;
	// cout << element << "\t" << x << "\t" << y << "\t" << z << endl;
	*element_ptr++ = element_number(element);
	*data_ptr++ = x;
	*data_ptr++ = y;
	*data_ptr++ = z;
      }

    sprintf(buf, "/%d", step);
    Group group (file.createGroup(buf));
    xout << grid(step, num, h5name.c_str());
    write_h5_float32(data, num, group, "data");
    free(data);
    write_h5_int8(elements, num, group, "element");
    free(elements);
    step++;
  }
  xout << end;

  return(0);
}

