/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | foam-extend: Open Source CFD
   \\    /   O peration     |
    \\  /    A nd           | For copyright notice see file Copyright
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of foam-extend.

    foam-extend is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your
    option) any later version.

    foam-extend is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
    General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with foam-extend.  If not, see <http://www.gnu.org/licenses/>.

Class
    pythonPalLinearElasticMisesPlastic

Description
    Mechanical law that uses pybind11 to call a trained Keras (TensorFlow)
    neural network mechanical law.

SourceFiles
    pythonPalLinearElasticMisesPlastic.C

Author
    Simon A. Rodriguez, UCD. All rights reserved
    Philip Cardiff, UCD. All rights reserved

\*---------------------------------------------------------------------------*/

#ifndef pythonPalLinearElasticMisesPlastic_H
#define pythonPalLinearElasticMisesPlastic_H

#include "mechanicalLaw.H"
#include "surfaceFields.H"


// #include <pybind11/embed.h> // everything needed for embedding
// #include <pybind11/numpy.h> //Support for numpy
// #include <vector>
// #include <pybind11/stl.h>
// #include <pybind11/eval.h>
// namespace py = pybind11;
// using namespace pybind11::literals;
#include "pythonPal.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
			 Class pythonPalLinearElasticMisesPlastic Declaration
\*---------------------------------------------------------------------------*/

class pythonPalLinearElasticMisesPlastic
:
    public mechanicalLaw
{
    // Private data

        pythonPal myPythonPal;

        //- Density
	    dimensionedScalar rho_;

    	//- Implicit stiffness used by the solid model
        //  Assuming convergence is achieved, this will not affect the final answer
        //  Setting it to the equivalent of 2*mu + lambda in elasticity is optimal
	    dimensionedScalar impK_;

        //- Total strain field: already stored in the mechanicalLaw base class
        //  (accessible with epsilon()) so it can be removed from here
        // volSymmTensorField epsilon_;

        //- List of states (history variables) for the machine learning model
        PtrList<volScalarField> states_;

        //- List of states calculated in the correct functions. The final one becomes
        // states_ for the machine learning model
        PtrList<volScalarField> dumStates_;

    // Private Member Functions

    	//- Disallow default bitwise copy construct
	    pythonPalLinearElasticMisesPlastic(const pythonPalLinearElasticMisesPlastic&);

    	//- Disallow default bitwise assignment
	    void operator=(const pythonPalLinearElasticMisesPlastic&);

    	//- Update the strain field: already stored in the mechanicalLaw base class
        //void updateStrain();      
        

public:

    //- Runtime type information
    TypeName("pythonPalLinearElasticMisesPlastic");

    // Constructors

	//- Construct from dictionary
	pythonPalLinearElasticMisesPlastic
	(
	    const word& name,
	    const fvMesh& mesh,
	    const dictionary& dict,
	    const nonLinearGeometry::nonLinearType& nonLinGeom
	);


    // Destructor

	virtual ~pythonPalLinearElasticMisesPlastic();


    // Member Functions

	//- Return density
	virtual tmp<volScalarField> rho() const;

	//- Return the implicit stiffness
	//  This is the diffusivity for the Laplacian term
	virtual tmp<volScalarField> impK() const;

	//- Return the bulk modulus
	virtual tmp<volScalarField> K() const;

	//- Calculate the stress
	virtual void correct(volSymmTensorField& sigma);

	//- Calculate the stress
	virtual void correct(surfaceSymmTensorField& sigma);

	//- Return material residual i.e. a measured of how convergence of
	//  the material model
	virtual scalar residual();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
