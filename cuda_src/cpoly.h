class CPoly {
public:
    int m_deg;
    int m_len;
    float * m_coefs;

    //basic constructor
    CPoly(float * coefs, int deg);

    //zero constructor
    CPoly(int deg);

    ~CPoly();

    void print_poly();
    void randomize(float min = -1, float max = 1);

    static CPoly * triv_mult(CPoly * a, CPoly * b);
    static CPoly * karatsuba(CPoly * a, CPoly * b);
    static bool compare(CPoly * a, CPoly * b);
};