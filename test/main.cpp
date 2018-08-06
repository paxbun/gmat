#include <gmat.h>
#include <string>
#include <iostream>
#include <vector>

using namespace std;

int foo()
{
	try
	{
		cout << "Addition and subtraction" << endl;
		cout << gmat({ { 1, 2, 3 } }) + gmat({ { 5, 6, 7 } }) << endl;
		cout << gmat({ { 1, 2, 3 } }) - gmat({ { 7, 6, 5 } }) << endl;
		cout << endl;
		
		cout << "Element multiplication and division" << endl;
		cout << gmat({ {1, 2, 3} }) * gmat({ {5, 6, 7} }) << endl;
		cout << gmat({ {1, 2, 3} }) / gmat({ {5, 6, 7} }) << endl;
		cout << endl;

		cout << "Matrix multiplication" << endl;
		cout <<
			gmat({
				{ 1, 2, 3, 4, },
				{ 5, 6, 7, 8, },
				{ 9, 10, 11, 12, },
				{ 13, 14, 15, 16, },
			}) % gmat({
				{ 1, 2, 3, },
				{ 4, 5, 6, },
				{ 7, 8, 9, },
				{ 10, 11, 12, },
			}) / 5.0
		<< endl;
		cout << endl;


	}
	catch (const gmat_exception & ex)
	{
		cout << "Exception: " << ex.what() << endl;
	}
	
	return 0;
}