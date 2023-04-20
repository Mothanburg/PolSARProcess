#include <Eigen/Eigen>
#include <iostream>
#include <complex>

using namespace Eigen;
using std::complex;

void refined_lee_filterC2(const ArrayXXd& c11, const ArrayXXd& c22, const ArrayXXd& c33,
    const ArrayXXcd& c12, const ArrayXXcd& c13, const ArrayXXcd& c23, int look)
{
    using Array77d = Array<double, 7, 7>;

    Index nr = c11.rows();
    Index na = c22.cols();
    ArrayXXd span_pad = ArrayXXd::Zero(nr + 6, na + 6);
    span_pad.block(3, 3, nr, na) = c11 + c22 + c33;

    Array33d w1;
    w1 << -1, 0, 1, -1, 0, 1, -1, 0, 1;
    Array33d w2;
    w2 << 0, 1, 1, -1, 0, 1, -1, -1, 0;
    Array33d w3;
    w3 << 1, 1, 0, 1, 0, -1, 0, -1, -1;
    Array33d w4;
    w4 << 1, 1, 0, 1, 0, -1, 0, -1, -1;

    Array77d pw1;
    pw1 <<
        0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1;
    Array77d pw2;
    pw2 <<
        1, 1, 1, 1, 1, 1, 1,
        0, 1, 1, 1, 1, 1, 1,
        0, 0, 1, 1, 1, 1, 1,
        0, 0, 0, 1, 1, 1, 1,
        0, 0, 0, 0, 1, 1, 1,
        0, 0, 0, 0, 0, 1, 1,
        0, 0, 0, 0, 0, 0, 1;
    Array77d pw3;
    pw3 <<
        1, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 0;
    Array77d pw4 = pw2.rowwise().reverse();
    Array77d pw5 = pw1.rowwise().reverse();
    Array77d pw6 = pw4.colwise().reverse();
    Array77d pw7 = pw3.colwise().reverse();
    Array77d pw8 = pw2.colwise().reverse();
    Array<Array77d, 8, 1> pws;
    pws << pw1, pw2, pw3, pw4, pw5, pw6, pw7, pw8;

    for(Index m = 0; m < nr; m++)
    {
        for (Index n = 0; n < na; n++)
        {
            Array77d window7x7 = span_pad.block(m, n, 7, 7);
            // 计算3x3平均矩阵
            Array33d mean3x3;
            for (Index i = 0; i < 3; i++)
            {
                for (Index j = 0; j < 3; j++)
                {
                    mean3x3(i, j) = window7x7.block(2 * i, 2 * j, 3, 3).mean();
                }
            }
            // 计算匹配的模板
            Array<double, 4, 1> tmp_arr1;
            tmp_arr1 << abs((mean3x3 * w1).sum()), abs((mean3x3 * w2).sum()), abs((mean3x3 * w3).sum()), abs((mean3x3 * w4).sum());
            Index I;
            tmp_arr1.maxCoeff(&I);
            double delta1, delta2;
            switch (I + 1)
            {
            case 1:
                delta1 = abs(mean3x3(1, 0) - mean3x3(1, 1));
                delta2 = abs(mean3x3(1, 2) - mean3x3(1, 1));
                if (delta1 > delta2)
                    I += 4;
                break;
            case 2:
                delta1 = abs(mean3x3(2, 0) - mean3x3(1, 1));
                delta2 = abs(mean3x3(0, 2) - mean3x3(1, 1));
                if (delta1 > delta2)
                    I += 4;
                break;
            case 3:
                delta1 = abs(mean3x3(2, 1) - mean3x3(1, 1));
                delta2 = abs(mean3x3(0, 1) - mean3x3(1, 1));
                if (delta1 > delta2)
                    I += 4;
                break;
            case 4:
                delta1 = abs(mean3x3(2, 2) - mean3x3(1, 1));
                delta2 = abs(mean3x3(0, 0) - mean3x3(1, 1));
                if (delta1 > delta2)
                    I += 4;
                break;
            default:
                break;
            }
            Array77d pw = pws(I);
            // 计算b
            Array77d z = window7x7 * pw;
            double z_mean = z.mean();
            double var_z = (z - z_mean).square().mean();
            double var_x = (var_z - z_mean * z_mean * look) / (1 + look);
            double b = var_x / var_z;
            // 计算协方差矩阵平均值
            Index r_start = 3 - m <= 0 ? 0 : 3 - m;
            Index m_start = m - 3 >= 0 ? m - 3 : 0;
            Index r_len = 7;
            if (m < 3)
                r_len = 4 + m;
            else if (m > nr - 4)
                r_len = nr - m + 3;
            Index c_start = 3 - n <= 0 ? 0 : 3 - n;
            Index n_start = n - 3 >= 0 ? n - 3 : 0;
            Index c_len = 7;
            if (n < 3)
                c_len = 4 + n;
            else if (n > na - 4)
                c_len = na - n + 3;

            double c11_mean = c11.block(m_start, n_start, r_len, c_len).cwiseProduct(pw.block(r_start, c_start, r_len, c_len)).mean();
            double c22_mean = c22.block(m_start, n_start, r_len, c_len).cwiseProduct(pw.block(r_start, c_start, r_len, c_len)).mean();
            double c33_mean = c33.block(m_start, n_start, r_len, c_len).cwiseProduct(pw.block(r_start, c_start, r_len, c_len)).mean();
            complex c12_mean = c12.block(m_start, n_start, r_len, c_len).cwiseProduct(pw.block(r_start, c_start, r_len, c_len)).mean();
            complex c13_mean = c13.block(m_start, n_start, r_len, c_len).cwiseProduct(pw.block(r_start, c_start, r_len, c_len)).mean();
            complex c23_mean = c23.block(m_start, n_start, r_len, c_len).cwiseProduct(pw.block(r_start, c_start, r_len, c_len)).mean();
            c11_mean + b * (c11(m, n) - c11_mean);
            c22_mean + b * (c22(m, n) - c22_mean);
            c33_mean + b * (c33(m, n) - c33_mean);
            c12_mean + b * (c12(m, n) - c12_mean);
            c13_mean + b * (c13(m, n) - c13_mean);
            c23_mean + b * (c23(m, n) - c23_mean);
        }
    }
}

int main()
{
    ArrayXXd c11 = ArrayXXd::Random(900, 1024);
    ArrayXXd c22 = ArrayXXd::Random(900, 1024);
    ArrayXXd c33 = ArrayXXd::Random(900, 1024);
    ArrayXXcd c12 = ArrayXXcd::Random(900, 1024);
    ArrayXXcd c13 = ArrayXXcd::Random(900, 1024);
    ArrayXXcd c23 = ArrayXXcd::Random(900, 1024);
    refined_lee_filterC2(c11, c22, c33, c12, c13, c23, 1);
    return 0;
}
