//我可以入选感动数院十大人物吗？之，矩阵计算器
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define MAX 50
#define EPS 1e-9

double A[MAX][MAX]; int aRow = 0, aCol = 0;
double B[MAX][MAX]; int bRow = 0, bCol = 0;
double C[MAX][MAX]; int cRow = 0, cCol = 0;

void PrintMatrix(double m[MAX][MAX], int row, int col);
void InputMatrix(double m[MAX][MAX], int *row, int *col, char name);
void Trans(double s[MAX][MAX], int sr, int sc, double d[MAX][MAX], int *dr, int *dc);
int  Multp(double x[MAX][MAX], int xr, int xc,
           double y[MAX][MAX], int yr, int yc, double z[MAX][MAX]);
int  Add(double x[MAX][MAX], int xr, int xc,
         double y[MAX][MAX], int yr, int yc, double z[MAX][MAX]);
int  Rank(double m[MAX][MAX], int r, int c);
int  Knee(double m[MAX][MAX], int n, double out[MAX][MAX]);
double Det(double m[MAX][MAX], int n);
void RowOp(double m[MAX][MAX], int r, int c);

void PrintMatrix(double m[MAX][MAX], int row, int col)
{
    int i, j;
    if (row == 0 || col == 0) { printf("空矩阵\n"); return; }
    for (i = 0; i < row; i++) {
        printf(" ");
        for (j = 0; j < col; j++) {
            double v = m[i][j];
            if (fabs(v) < EPS) v = 0.0;
            printf("%11.4f", v);
        }
        printf("\n");
    }
}

void InputMatrix(double m[MAX][MAX], int *row, int *col, char name)
{
    int i, j;
    printf("输入矩阵 %c 的行数和列数: ", name);
    scanf("%d %d", row, col);
    if (*row < 1 || *col < 1 || *row > MAX || *col > MAX) {
        printf("尺寸不合法！(1~%d)\n", MAX);
        *row = *col = 0;
        return;
    }
    printf("按行输入 %d x %d 个元素:\n", *row, *col);
    for (i = 0; i < *row; i++)
        for (j = 0; j < *col; j++)
            scanf("%lf", &m[i][j]);
    printf("矩阵 %c 输入完成！\n", name);
}

void Trans(double s[MAX][MAX], int sr, int sc, double d[MAX][MAX], int *dr, int *dc)
{
    int i, j;
    for (i = 0; i < sr; i++)
        for (j = 0; j < sc; j++)
            d[j][i] = s[i][j];
    *dr = sc; *dc = sr;
}

int Multp(double x[MAX][MAX], int xr, int xc,
          double y[MAX][MAX], int yr, int yc, double z[MAX][MAX])
{
    int i, j, k;
    if (xc != yr) return 0;
    for (i = 0; i < xr; i++)
        for (j = 0; j < yc; j++) {
            z[i][j] = 0.0;
            for (k = 0; k < xc; k++)
                z[i][j] += x[i][k] * y[k][j];
        }
    return 1;
}

int Add(double x[MAX][MAX], int xr, int xc,
        double y[MAX][MAX], int yr, int yc, double z[MAX][MAX])
{
    int i, j;
    if (xr != yr || xc != yc) return 0;
    for (i = 0; i < xr; i++)
        for (j = 0; j < xc; j++)
            z[i][j] = x[i][j] + y[i][j];
    return 1;
}

//高斯！赐予我力量吧！
int Gauss(double t[MAX][MAX], int r, int c, int *swapCnt)
{
    int i, j, k, piv, row = 0;
    *swapCnt = 0;
    for (j = 0; j < c && row < r; j++) {
        piv = -1;
        for (i = row; i < r; i++)
            if (fabs(t[i][j]) > EPS) { piv = i; break; }
        if (piv == -1) continue;
        if (piv != row) {
            for (k = 0; k < c; k++) {
                double tmp = t[row][k];
                t[row][k] = t[piv][k];
                t[piv][k] = tmp;
            }
            (*swapCnt)++;
        }
        for (i = row + 1; i < r; i++) {
            double f = t[i][j] / t[row][j];
            for (k = j; k < c; k++)
                t[i][k] -= f * t[row][k];
        }
        row++;
    }
    return row;
}
int Rank(double m[MAX][MAX], int r, int c)
{
    double t[MAX][MAX];
    int i, j, sw;
    for (i = 0; i < r; i++)
        for (j = 0; j < c; j++)
            t[i][j] = m[i][j];
    return Gauss(t, r, c, &sw);
}

int Knee(double m[MAX][MAX], int n, double out[MAX][MAX])  // 逆！
{
    double t[MAX][2 * MAX];
    int i, j, k, piv;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++) {
            t[i][j] = m[i][j];
            t[i][n + j] = (i == j) ? 1.0 : 0.0;
        }
    for (j = 0; j < n; j++) {
        piv = j;
        for (i = j + 1; i < n; i++)
            if (fabs(t[i][j]) > fabs(t[piv][j])) piv = i;
        if (fabs(t[piv][j]) < EPS) return 0;  // 奇异，多么没逆
        if (piv != j)
            for (k = 0; k < 2 * n; k++) {
                double tmp = t[j][k];
                t[j][k] = t[piv][k];
                t[piv][k] = tmp;
            }
        for (i = 0; i < n; i++) {
            if (i == j) continue;
            double f = t[i][j] / t[j][j];
            for (k = 0; k < 2 * n; k++)
                t[i][k] -= f * t[j][k];
        }
    }
    for (i = 0; i < n; i++) {
        double f = t[i][i];
        for (k = 0; k < 2 * n; k++)
            t[i][k] /= f;
        for (j = 0; j < n; j++)
            out[i][j] = t[i][n + j];
    }
    return 1;
}

double Det(double m[MAX][MAX], int n)
{
    double t[MAX][MAX], det = 1.0;
    int i, j, k, piv, sw = 0;
    for (i = 0; i < n; i++)
        for (j = 0; j < n; j++)
            t[i][j] = m[i][j];
    for (j = 0; j < n; j++) {
        piv = j;
        for (i = j + 1; i < n; i++)
            if (fabs(t[i][j]) > fabs(t[piv][j])) piv = i;
        if (fabs(t[piv][j]) < EPS) return 0.0;
        if (piv != j) {
            for (k = 0; k < n; k++) {
                double tmp = t[j][k];
                t[j][k] = t[piv][k];
                t[piv][k] = tmp;
            }
            sw++;
        }
        for (i = j + 1; i < n; i++) {
            double f = t[i][j] / t[j][j];
            for (k = j; k < n; k++)
                t[i][k] -= f * t[j][k];
        }
    }
    for (i = 0; i < n; i++) det *= t[i][i];
    return (sw % 2) ? -det : det;
}

void RowOp(double m[MAX][MAX], int r, int c)
{
    int t, i, j, p;
    double f, k;
    printf("1.交换两行 2.某行乘k 3.某行乘k加到另一行\n请选择: ");
    scanf("%d", &t);
    if (t == 1) {
        printf("交换第几行和第几行(1~%d): ", r);
        scanf("%d %d", &i, &j);
        i--; j--;
        if (i < 0 || j < 0 || i >= r || j >= r) return;
        for (p = 0; p < c; p++) {
            double tmp = m[i][p];
            m[i][p] = m[j][p];
            m[j][p] = tmp;
        }
    } else if (t == 2) {
        printf("第几行(1~%d), 乘数k: ", r);
        scanf("%d %lf", &i, &k);
        i--;
        if (i < 0 || i >= r) return;
        for (p = 0; p < c; p++) m[i][p] *= k;
    } else if (t == 3) {
        printf("第几行乘k, 加到第几行: ");
        scanf("%d %lf %d", &i, &k, &j);
        i--; j--;
        if (i < 0 || j < 0 || i >= r || j >= r) return;
        for (p = 0; p < c; p++) m[j][p] += k * m[i][p];
    } else return;
    printf("操作后的矩阵:\n");
    PrintMatrix(m, r, c);
}

int main(void)
{
system("chcp 65001 > nul");
int choice;
while (1) {
printf("欢迎:D");
printf("\n===== 矩阵计算器 =====\n");
printf("   1. 输入/修改矩阵 A\n");
printf("   2. 输入/修改矩阵 B\n");
printf("   3. 查看矩阵 A\n");
printf("   4. 查看矩阵 B\n");
printf("   5. 查看矩阵 C\n");
printf("   6. C = A 的转置\n");
printf("   7. C = B 的转置\n");
printf("   8. C = A x B   （矩阵乘法）\n");
printf("   9. C = A + B\n");
printf("  10. 求 A 的秩\n");
printf("  11. C = A 的逆\n");
printf("  12. 对 A 做初等行变换\n");
printf("  13. 求 A 的行列式\n");
printf("   0. 退出\n");
printf("请选择: ");
if (scanf("%d", &choice) != 1) {
printf("输入错误，退出。\n");
break;
}


    switch (choice) {
    case 0:
        printf("再见！\n");
        return 0;

    case 1:
        InputMatrix(A, &aRow, &aCol, 'A');
        if (aRow > 0) PrintMatrix(A, aRow, aCol);
        break;

    case 2:
        InputMatrix(B, &bRow, &bCol, 'B');
        if (bRow > 0) PrintMatrix(B, bRow, bCol);
        break;

    case 3:
        printf("A (%d x %d):\n", aRow, aCol);
        PrintMatrix(A, aRow, aCol);
        break;

    case 4:
        printf("B (%d x %d):\n", bRow, bCol);
        PrintMatrix(B, bRow, bCol);
        break;

    case 5:
        printf("C (%d x %d):\n", cRow, cCol);
        PrintMatrix(C, cRow, cCol);
        break;

    case 6:
        if (aRow == 0) { printf("A 是空的\n"); break; }
        Trans(A, aRow, aCol, C, &cRow, &cCol);
        printf("C = A 的转置:\n");
        PrintMatrix(C, cRow, cCol);
        break;

    case 7:
        if (bRow == 0) { printf("B 是空的\n"); break; }
        Trans(B, bRow, bCol, C, &cRow, &cCol);
        printf("C = B 的转置:\n");
        PrintMatrix(C, cRow, cCol);
        break;

    case 8:
        if (aRow == 0 || bRow == 0) { printf("A 或 B 是空的\n"); break; }
        if (Multp(A, aRow, aCol, B, bRow, bCol, C)) {
            cRow = aRow; cCol = bCol;
            printf("C = A x B:\n");
            PrintMatrix(C, cRow, cCol);
        } else {
            printf("维度不匹配：A 的列数(%d) 必须等于 B 的行数(%d)\n",
                   aCol, bRow);
        }
        break;

    case 9:
        if (aRow == 0 || bRow == 0) { printf("A 或 B 是空的\n"); break; }
        if (Add(A, aRow, aCol, B, bRow, bCol, C)) {
            cRow = aRow; cCol = aCol;
            printf("C = A + B:\n");
            PrintMatrix(C, cRow, cCol);
        } else {
            printf("维度不匹配：A 是 %dx%d，B 是 %dx%d，必须相同\n",
                   aRow, aCol, bRow, bCol);
        }
        break;

    case 10:
        if (aRow == 0) { printf("A 是空的\n"); break; }
        printf("rank(A) = %d\n", Rank(A, aRow, aCol));
        break;

    case 11:
        if (aRow == 0) { printf("A 是空的\n"); break; }
        if (aRow != aCol) { printf("只有方阵才能求逆\n"); break; }
        if (Knee(A, aRow, C)) {
            cRow = aRow; cCol = aCol;
            printf("C = A 的逆:\n");
            PrintMatrix(C, cRow, cCol);
        } else {
            printf("A 不可逆（奇异矩阵）\n");
        }
        break;

    case 12:
        if (aRow == 0) { printf("A 是空的\n"); break; }
        RowOp(A, aRow, aCol);
        break;

    case 13:
        if (aRow == 0) { printf("A 是空的\n"); break; }
        if (aRow != aCol) { printf("只有方阵才有行列式\n"); break; }
        printf("det(A) = %.6f\n", Det(A, aRow));
        break;

    default:
        printf("没有这个选项\n");
    }
}
return 0;
}