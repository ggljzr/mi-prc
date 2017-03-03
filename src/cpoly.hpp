class CPoly {
public:
    int m_deg;
    int m_len;
    double * m_coefs;

    //basic constructor
    CPoly(double * coefs, int deg);

    //zero constructor
    CPoly(int deg);

    ~CPoly();

    void print_poly();

    static CPoly * triv_mult(CPoly * a, CPoly * b);
};