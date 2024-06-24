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

\*---------------------------------------------------------------------------*/

#include "pythonPalLinearElasticMisesPlastic.H"
#include "addToRunTimeSelectionTable.H"
#include "zeroGradientFvPatchFields.H"


#include <chrono>
using namespace std::chrono;

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
    defineTypeNameAndDebug(pythonPalLinearElasticMisesPlastic, 0);
    addToRunTimeSelectionTable
    (
        mechanicalLaw, pythonPalLinearElasticMisesPlastic, linGeomMechLaw
    );
}


// * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * * * //

// void Foam::pythonPalLinearElasticMisesPlastic::updateStrain()
// {
//     if (incremental())
//     {
//         // Lookup gradient of displacement increment
//         const volTensorField& gradDD =
//             mesh().lookupObject<volTensorField>("grad(DD)");

//         // Calculate the total strain
//         epsilon_ = epsilon_.oldTime() + symm(gradDD);
//     }
//     else
//     {
//         // Lookup gradient of displacement
//         const volTensorField& gradD =
//             mesh().lookupObject<volTensorField>("grad(D)");

//         // Calculate the total strain
//         epsilon_ = symm(gradD);
//     }
// }

// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

// Construct from dictionary
Foam::pythonPalLinearElasticMisesPlastic::pythonPalLinearElasticMisesPlastic
(
    const word& name,
    const fvMesh& mesh,
    const dictionary& dict,
    const nonLinearGeometry::nonLinearType& nonLinGeom
)
:
    mechanicalLaw(name, mesh, dict, nonLinGeom),
    // Note: you need to initialise all data members here, and in the correct
    // order
    // myPythonPal("python_code.py", false),
    myPythonPal("python_code.py", false),
    rho_(dict.lookup("rho")),
    impK_(dict.lookup("implicitStiffness")),
    // epsilon_
    // (
    //     IOobject
    //     (
    //         "epsilon",
    //         mesh.time().timeName(),
    //         mesh,
    //         IOobject::NO_READ,
    //         IOobject::NO_WRITE
    //     ),
    //     mesh,
    //     dimensionedSymmTensor("zero", dimless, symmTensor::zero)
    // ),
    // states_(dict.lookupOrDefault<int>("nStates", 20)),
    // dumStates_(dict.lookupOrDefault<int>("nStates", 20))
    states_(dict.lookupOrDefault<int>("nStates", 10)),
    dumStates_(dict.lookupOrDefault<int>("nStates", 10))
    // states_(dict.lookupOrDefault<int>("nStates", 15)),
    // dumStates_(dict.lookupOrDefault<int>("nStates", 15))
    // states_(dict.lookupOrDefault<int>("nStates", 20)),
    // dumStates_(dict.lookupOrDefault<int>("nStates", 20))
    
{

    // Initialise the state fields
    Info<< "Initialising " << states_.size() << " states" << endl;
    forAll(states_, stateI)
    {
        const word stateNameI = "state_" + Foam::name(stateI);
        Info<< "Creating state: " << stateNameI << endl;
        states_.set
        (
            stateI,
            new volScalarField
            (
                IOobject
                (
                    stateNameI,
                    mesh.time().timeName(),
                    mesh,
                    // IOobject::NO_READ,
                    // IOobject::AUTO_WRITE
                    IOobject::READ_IF_PRESENT,
                    IOobject::AUTO_WRITE
                ),
                mesh,
                dimensionedScalar("zero", dimless, 0.0)
                // dimensionedScalar("zero", dimless, stateI)
            )
        );

        // Info << states_[stateI].oldTime(); 

        // Tell each state to store its old time
        states_[stateI].storeOldTime();
    }

    forAll(dumStates_, stateI)
    {
        const word stateNameI = "dumStates_" + Foam::name(stateI);
        Info<< "Creating dumStates_: " << stateNameI << endl;
        dumStates_.set
        (
            stateI,
            new volScalarField
            (
                IOobject
                (
                    stateNameI,
                    mesh.time().timeName(),
                    mesh,
                    IOobject::READ_IF_PRESENT,
                    IOobject::AUTO_WRITE
                    // IOobject::NO_READ,
                    // IOobject::AUTO_WRITE
                ),
                mesh,
                dimensionedScalar("zero", dimless, 0.0)
                // dimensionedScalar("zero", dimless, stateI)
            )
        );

        // Tell each state to store its old time
        dumStates_[stateI].storeOldTime();
    }

    // Info << "epsilon at t " << endl << epsilon() << endl;

    // Info << "epsilon old " << endl << epsilon().oldTime() << endl;

    // Load the python file and evaluate it
    const word pythonMod =
        dict.lookupOrDefault<word>("pythonModule", "python_code.py");
    pythonPal myPythonPal(pythonMod, false);


    scalar nStates = states_.size(); //This should a class attribute?
    Info << "nStates is " << nStates << endl;

    myPythonPal.passScalarToPython(nStates, "nStates");
    // myPythonPal.execute();

    // Check impK_ is positive
    if (impK_.value() < SMALL)
    {
        FatalErrorIn
        (
            "Foam::pythonPalLinearElasticMisesPlastic::pythonPalLinearElasticMisesPlastic\n"
            "(\n"
            "    const word& name,\n"
            "    const fvMesh& mesh,\n"
            "    const dictionary& dict\n"
            ")"
        )   << "The implicitStiffness should be positive!"
            << abort(FatalError);
    }

    // Store the old time
    epsilon().oldTime();
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::pythonPalLinearElasticMisesPlastic::~pythonPalLinearElasticMisesPlastic()
{}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

Foam::tmp<Foam::volScalarField> Foam::pythonPalLinearElasticMisesPlastic::rho() const
{
    tmp<volScalarField> tresult
    (
        new volScalarField
        (
            IOobject
            (
                "rho",
                mesh().time().timeName(),
                mesh(),
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh(),
            rho_,
            zeroGradientFvPatchScalarField::typeName
        )
    );

#ifdef OPENFOAMESIORFOUNDATION
    tresult.ref().correctBoundaryConditions();
#else
    tresult().correctBoundaryConditions();
#endif

    return tresult;
}


Foam::tmp<Foam::volScalarField> Foam::pythonPalLinearElasticMisesPlastic::impK() const
{
    return tmp<volScalarField>
    (
        new volScalarField
        (
            IOobject
            (
                "impK",
                mesh().time().timeName(),
                mesh(),
                IOobject::NO_READ,
                IOobject::NO_WRITE
            ),
            mesh(),
            impK_
        )
    );
}

Foam::tmp<Foam::volScalarField> Foam::pythonPalLinearElasticMisesPlastic::K() const
{
    notImplemented("Foam::pythonPalLinearElasticMisesPlastic::K()");

    // Keep the compiler happy
    return impK();
}



void Foam::pythonPalLinearElasticMisesPlastic::correct(volSymmTensorField& sigma)
{
    // Update total strain
    updateEpsilon();


    // Info << "epsilon at t " << endl << epsilon() << endl;

    // Info << "epsilon old " << endl << epsilon().oldTime() << endl;

    // myPythonPal.execute("exit()");

    myPythonPal.passScalarToPython(int(mesh().time().value()), "time"); 

    // myPythonPal.execute("time = 45");

    // myPythonPal.execute("print('Simon Rodriguez time is ' + str(time))");

    // myPythonPal.execute("print(dir())");
    
    // myPythonPal.execute("print(predict)");

    // myPythonPal.execute("exit()");

    // Info << "Time is "<< mesh().time().value() << endl << endl;

    // Calculate strain increment field
    // It's not declared as const because It will be sent to Python, even though 
    // It will not be modified
    volSymmTensorField DEpsilon = epsilon() - epsilon().oldTime();

    // Pass sigma, epsilon and DEpsilon internal fields to Python side
    myPythonPal.passToPython(sigma, "sigma");
    myPythonPal.passToPython(epsilon(), "epsilon");
    myPythonPal.passToPython(DEpsilon, "DEpsilon");
    // myPythonPal.passToPython(states_, "states");

    // Pass the states
    forAll(states_, stateI) 
    {
        myPythonPal.passToPython(states_[stateI], "states" + std::to_string(stateI));
        myPythonPal.passToPython(dumStates_[stateI], "dumStates" + std::to_string(stateI));
    }
    // myPythonPal.execute("statesSoFar()");
    myPythonPal.execute("predict()");

    //Now for the boundary field
    
        forAll(sigma.boundaryField(), patchI)
        {
            // Info<< "patchI is" << patchI << endl << endl;

            if (sigma.boundaryField()[patchI].size() != 0)
            {
                myPythonPal.passToPython(sigma.boundaryFieldRef()[patchI], "sigma");
                myPythonPal.passToPython(epsilon().boundaryFieldRef()[patchI], "epsilon");
                myPythonPal.passToPython(DEpsilon.boundaryFieldRef()[patchI], "DEpsilon");
            
                // Pass the states
                forAll(states_, stateI) 
                {
                    // Info<<" states + stateI is " <<   "states" + std::to_string(stateI)  << endl;
                    myPythonPal.passToPython(states_[stateI].boundaryFieldRef()[patchI], "states" + std::to_string(stateI));
                    myPythonPal.passToPython(dumStates_[stateI].boundaryFieldRef()[patchI], "dumStates" + std::to_string(stateI));
                }
            }

        myPythonPal.execute("predict()");           
        }
    
    // myPythonPal.execute("predict()");   





    

    // forAll(states_, stateI) 
    // {

    //     forAll(states_[stateI].boundaryField(), patchI)
    //     {
    //         if (states_[stateI].boundaryField()[patchI].size() != 0) 
    //         {
    //             myPythonPal.passToPython(sigma, "sigma");
    //             myPythonPal.passToPython(epsilon(), "epsilon");
    //             myPythonPal.passToPython(DEpsilon, "DEpsilon");
    //             myPythonPal.passToPython(states_[stateI], "states" + std::to_string(stateI));
    //             myPythonPal.passToPython(dumStates_[stateI], "dumStates" + std::to_string(stateI));
    //         }

    //     }
    // }

    // myPythonPal.execute("predict()");    

    // DEpsilon.write();

}




void Foam::pythonPalLinearElasticMisesPlastic::correct(surfaceSymmTensorField& sigma)
{
    notImplemented
    (
        "void Foam::pythonPalLinearElasticMisesPlastic::correct(surfaceSymmTensorField&)"
    );
}


Foam::scalar Foam::pythonPalLinearElasticMisesPlastic::residual()
{
    // We can probably calculate this in some way so that the solid model
    // knows if the mechanical law has converged
    return 0.0;
}


// ************************************************************************* //
