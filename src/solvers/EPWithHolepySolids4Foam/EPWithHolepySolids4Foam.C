/* License
    This program is part of pythonPal4Foam.
    This program is free software: you can redistribute it and/or modify 
    it under the terms of the GNU General Public License as published 
    by the Free Software Foundation, either version 3 of the License, 
    or (at your option) any later version.
    This program is distributed in the hope that it will be useful, 
    but WITHOUT ANY WARRANTY; without even the implied warranty of 
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
    See the GNU General Public License for more details. You should have 
    received a copy of the GNU General Public License along with this 
    program. If not, see <https://www.gnu.org/licenses/>. 
   Application
    pythonsolids4Foam
   Original solver
    solids4Foam
   Modified by
    Simon A. Rodriguez, UCD. All rights reserved
    Philip Cardiff, UCD. All rights reserved
   Description
    General solver where the solved mathematical model (fluid, solid or
    fluid-solid) is chosen at run-time.
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "physicsModel.H"
#include "pythonPal.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
// #   include "setRootCase.H"
#   include "setRootCase2.H"
#   include "createTime.H"
#   include "solids4FoamWriteHeader.H"

    // pythonPal myPythonPal("python_script.py", true);
    pythonPal myPythonPal("python_code.py", true);

    // Create the general physics class
    autoPtr<physicsModel> physics = physicsModel::New(runTime);

    while (runTime.run())
    {
        physics().setDeltaT(runTime);

        runTime++;

        Info<< "Time = " << runTime.timeName() << nl << endl;

        // myPythonPal.passScalarToPython(runTime.value(), "time"); 

        // Solve the mathematical model
        physics().evolve();

        // Let the physics model know the end of the time-step has been reached
        physics().updateTotalFields();

        if (runTime.outputTime())
        {
            // physics().writeFields(runTime);

        // myPythonPal.passScalarToPython(runTime.value(), "time"); 

        const fvMesh& mesh = runTime.lookupObject<fvMesh>("region0");


        volSymmTensorField epsilon = mesh.lookupObject<volSymmTensorField>("epsilon_");
        volSymmTensorField sigma = mesh.lookupObject<volSymmTensorField>("sigma");
        volVectorField D = mesh.lookupObject<volVectorField>("D");
        // volVectorField DEpsilon = mesh.lookupObject<volVectorField>("DEpsilon");
        // DEpsilon.write();

        // for (int i = 0; i < 20; i++)
        for (int i = 0; i < 10; i++)
        // for (int i = 0; i < 15; i++)
        // for (int i = 0; i < 15; i++)
        {
            //Retrieve the states from the mechanical law
            word tempWord= "state_" + std::to_string(i);
            word tempDumWord= "dumStates_" + std::to_string(i);
            volScalarField& tempState = mesh.lookupObjectRef<volScalarField>(tempWord);
            volScalarField& tempDumStates = mesh.lookupObjectRef<volScalarField>(tempDumWord);

            tempState = tempDumStates;

            //Pass the states to Python side

            // myPythonPal.passToPython();
            


        }

            physics().writeFields(runTime);

        myPythonPal.passToPython(epsilon, "epsilon");
        myPythonPal.passToPython(sigma, "sigma");
        myPythonPal.passToPython(D, "D");

        // Get references to the boundaries
        label Id_Up = mesh.boundaryMesh().findPatchID("up");
        label Id_Down = mesh.boundaryMesh().findPatchID("down");
        label Id_Right = mesh.boundaryMesh().findPatchID("right");
        label Id_Hole = mesh.boundaryMesh().findPatchID("hole");

        vectorField& D_Right = D.boundaryFieldRef()[Id_Right]; 
        vectorField& D_Up = D.boundaryFieldRef()[Id_Up];
        vectorField& D_Down = D.boundaryFieldRef()[Id_Down];
        vectorField& D_Hole = D.boundaryFieldRef()[Id_Hole];

        // D_Right = D.boundaryField()[Id_Right];
        // D_Up = D.boundaryField()[Id_Up];
        // D_Down = D.boundaryField()[Id_Down];

        symmTensorField epsilon_Right = epsilon.boundaryFieldRef()[Id_Right];
        symmTensorField epsilon_Down = epsilon.boundaryFieldRef()[Id_Down];
        symmTensorField epsilon_Up = epsilon.boundaryFieldRef()[Id_Up];
        symmTensorField epsilon_Hole = epsilon.boundaryFieldRef()[Id_Hole];

        symmTensorField sigma_Right = sigma.boundaryFieldRef()[Id_Right];
        symmTensorField sigma_Down = sigma.boundaryFieldRef()[Id_Down];
        symmTensorField sigma_Up = sigma.boundaryFieldRef()[Id_Up];
        symmTensorField sigma_Hole = sigma.boundaryFieldRef()[Id_Hole];

        myPythonPal.passToPython(D_Right, "D_Right");
        myPythonPal.passToPython(D_Down, "D_Down");
        myPythonPal.passToPython(D_Up, "D_Up");
        myPythonPal.passToPython(D_Hole, "D_Hole");
        myPythonPal.passToPython(epsilon_Right, "epsilon_Right");
        myPythonPal.passToPython(epsilon_Down, "epsilon_Down");
        myPythonPal.passToPython(epsilon_Up, "epsilon_Up");
        myPythonPal.passToPython(epsilon_Hole, "epsilon_Hole");
        myPythonPal.passToPython(sigma_Right, "sigma_Right");
        myPythonPal.passToPython(sigma_Down, "sigma_Down");
        myPythonPal.passToPython(sigma_Up, "sigma_Up");
        myPythonPal.passToPython(sigma_Hole, "sigma_Hole");

        // Info << "epsilon is  " << epsilon << endl;


        myPythonPal.execute("serialise_fields()");

        }

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    physics().end();

    Info<< nl << "End" << nl << endl;

    return(0);
}


// ************************************************************************* //