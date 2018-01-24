#pragma once
#include <math.h>
#include <vector>
#include <sstream>
#include <iostream>

template< typename D >
std::string vecToString( std::vector< D > toS )
{
	std::stringstream alf;
	alf << "{ ";
	for ( int i = 0; i < toS.size() - 1; i++ )
	{
		alf << toS[ i ] << ", ";
	}
	alf << toS[ toS.size() - 1 ] << " }";
	return alf.str();
}

// sigmoid
template< typename D >
D squash( D de_squashed )
{
	return 1.0 / ( 1.0 + exp( -de_squashed ) );
}

// pair< < vec >, < vec< vec > > >
template< typename D >
std::vector< D > dot( std::pair< std::vector< D >, std::vector< std::vector< D >>> M )
{
	std::vector< D > vd;
	for ( int k = 0; k < M.second[ 0 ].size(); k++ )
	{
		vd.push_back( 0.0 );
		for ( int j = 0; j < M.first.size(); j++ )
		{
			vd[ k ] += M.first[ j ] * M.second[ j ][ k ];
		}
	}
	return vd;
}

// subtract two vectors element wise
//if different sizes returns first
template< typename D >
std::vector< D > elemWiseSubtraction( std::vector< D > a, std::vector< D > b )
{
	if ( a.size() != b.size() ) {
		std::cout << a.size() << " nope " << b.size() << std::endl;
		return a;
	}
	std::vector<D> c;
	for ( int i = 0; i < a.size(); i++ ) {
		c.push_back( a[ i ] - b[ i ] );
	}
	return c;
}

// Squared Error
//if different sizes returns first vector
template< typename D >
D sumSquaredDifference( std::vector< D > a, std::vector< D > b )
{
	if ( a.size() != b.size() ) {
		std::cout << a.size() << " nope " << b.size() << std::endl;
		return pow(a[0] - b[0], 2);
	}

	D c = D(0.0);
	for ( int i = 0; i < a.size(); i++ )
	{
		c += pow( a[ i ] - b[ i ], 2 );
	}
	return c;
}

template< typename D >
D average(std::vector< D > a)
{
	D c = D(0.0);
	for (int i = 0; i < a.size(); i++)
	{
		c += a[i];
	}
	c /= a.size();
	return c;
}