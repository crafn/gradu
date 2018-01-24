/* Adapted from: */
/* Z_2 lattice gauge simulation */
/* Michael Creutz <creutz@bnl.gov>     */
/* http://thy.phy.bnl.gov/~creutz/z2.c */

extern int rand();
extern int srand(unsigned int seed);
extern int printf(const char* fmt, ..);
extern double exp(double e);

/* the lattice is of dimensions SIZE**4  */
const int SIZE = 10;
typedef int field(5) Links; // Last index is link direction
int field(5) link;

const int RAND_DATA_COUNT = 128;
// Poor man's random generator
const float rand_data[RAND_DATA_COUNT] = {
	0.7657781957, 0.3805083251, 0.9761230163, 0.0479723098, 0.0279490484, 0.4931316109, 0.1450683465, 0.9376585919,
	0.6884426611, 0.3170458978, 0.8036460290, 0.9177376412, 0.5139127222, 0.3637061834, 0.1372744120, 0.6666601035,
	0.2500191582, 0.6222415203, 0.0212470260, 0.4068253360, 0.7077081975, 0.8562926318, 0.9476930496, 0.2077958577,
	0.3629353587, 0.9022419999, 0.4279602440, 0.7047110816, 0.6137632899, 0.6602612138, 0.3782546900, 0.6549581081,
	0.9369041805, 0.6833423437, 0.8913837099, 0.2998812058, 0.0645599599, 0.3005034792, 0.5727744059, 0.1326779427,
	0.1322915428, 0.4387057023, 0.5945456798, 0.8373149732, 0.1804350690, 0.2150155768, 0.7268310698, 0.7671266682,
	0.5564608541, 0.8607236996, 0.1322729299, 0.2886790577, 0.0011318485, 0.9463158670, 0.7408911077, 0.5023065840,
	0.1891468385, 0.6097329492, 0.7166870644, 0.0981457291, 0.6509897858, 0.4763264338, 0.9583957865, 0.4588363753,
	0.8344191634, 0.8760433369, 0.8208734982, 0.4331271974, 0.8005440855, 0.9397875463, 0.7418332230, 0.9054539493,
	0.7969143115, 0.5675450794, 0.0541708636, 0.3334961511, 0.2479674044, 0.8801757947, 0.7605885882, 0.7697549422,
	0.0110490305, 0.3614833868, 0.8291620364, 0.2281247044, 0.5728353375, 0.8549794528, 0.0701704805, 0.7598098185,
	0.0222724539, 0.4779942366, 0.0145280567, 0.9913344734, 0.3142968539, 0.9400281562, 0.2356176665, 0.8406913590,
	0.8822662537, 0.8401935972, 0.9853635334, 0.7133342380, 0.6976501531, 0.0905732346, 0.2622729663, 0.5345999571,
	0.7619733939, 0.1469705382, 0.6678419412, 0.0691592161, 0.1022249254, 0.9824920183, 0.9332599092, 0.4412841027,
	0.1498436465, 0.0394898997, 0.5205901099, 0.0715307497, 0.1417755629, 0.7016223694, 0.2137731064, 0.7178876939,
	0.6215236465, 0.2859837641, 0.4424311934, 0.4714369466, 0.1979122500, 0.3146549199, 0.4962738184, 0.8967943074,
};

void moveup(int matrix(5) *x, int d)
{
	x(d) += 1;
	if (x(d) >= SIZE) x(d) -= SIZE; 
}

void movedown(int matrix(5) *x, int d)
{
	x(d) -= 1;
	if (x(d) < 0) x(d) += SIZE;
}

void coldstart()
{
	parallel (link; link) {
		link(id(0), id(1), id(2), id(3), id(4)) = 1;
	}
}

double update(double beta, int iter)
{
	float action = 0.0; 

	parallel (link; link; oddeven) {
		int dperp;
		float staplesum = 0;
		int staple;
		float bplus;
		float bminus;

		int d = id(4);
		for (dperp = 0; dperp < 4; dperp += 1) {
			if (dperp != d) {
				movedown(&id, dperp);
				int v1 = link(id(0), id(1), id(2), id(3), dperp);
				int v2 = link(id(0), id(1), id(2), id(3), d);
				staple = v1*v2;
				moveup(&id, d);
				staple *= link(id(0), id(1), id(2), id(3), dperp);
				moveup(&id, dperp);
				staplesum += staple;

				staple = link(id(0), id(1), id(2), id(3), dperp);
				moveup(&id, dperp);
				movedown(&id, d);
				staple *= link(id(0), id(1), id(2), id(3), d);
				movedown(&id, dperp);
				staple *= link(id(0), id(1), id(2), id(3), dperp);
				staplesum += staple;
			}
		}

		bplus = exp(beta*staplesum);
		bminus = 1/bplus;
		bplus = bplus/(bplus+bminus);

		int rand_ix = id(0) + id(1)*SIZE + id(3)*SIZE*SIZE + id(4)*SIZE*SIZE*SIZE + iter*SIZE*SIZE*SIZE*SIZE;
		if (rand_data[rand_ix % RAND_DATA_COUNT] < bplus) {
			link(id(0), id(1), id(2), id(3), d) = 1;
			action += staplesum;
		} else {
			link(id(0), id(1), id(2), id(3), d) = -1;
			action -= staplesum;
		}
	}

	action /= SIZE*SIZE*SIZE*SIZE*4*6;
	return 1.0 - action;
}

int main()
{
	link = alloc_device_field(SIZE, SIZE, SIZE, SIZE, 4);

	double beta;
	double action;
	double dbeta = 0.01;

	coldstart();

	int iter = 0;
	for (beta = 1; beta > 0.0; beta -= dbeta) {
		action = update(beta, iter);
		printf("%g\t%g\n", beta, action); 
		++iter;
	}
	printf("\n\n");
	for (beta = 0; beta < 1.0; beta += dbeta) {
		action = update(beta, iter);
		printf("%g\t%g\n", beta, action); 
		++iter;
	}

	free_device_field(link);

	return 0;
}

