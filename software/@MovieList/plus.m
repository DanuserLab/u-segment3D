function C = plus(A,B,outputDirectory)
% Override plus (+) operation to create a unique union of two MovieLists
%
% Use the plus function explicitly to specify an output directory other
% than pwd
%
% See also plus, unique
%
% Copyright (C) 2024, Danuser Lab - UTSouthwestern 
%
% This file is part of uSegment3D_Package.
% 
% uSegment3D_Package is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% (at your option) any later version.
% 
% uSegment3D_Package is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
% 
% You should have received a copy of the GNU General Public License
% along with uSegment3D_Package.  If not, see <http://www.gnu.org/licenses/>.
% 
% 

% Mark Kittisopikul, March 2018
% Goldman Lab
% Northwestern University

    if(nargin < 3)
        outputDirectory = pwd;
    end
    A_isMovieList = isa(A,'MovieList');
    A_isMovieData = isa(A,'MovieData');

    B_isMovieList = isa(B,'MovieList');
    B_isMovieData = isa(B,'MovieData');

    assert(A_isMovieList || B_isMovieList);

    if(A_isMovieData)
        A = MovieList(A);
    end
    if(B_isMovieData)
        B = MovieList(B);
    end
    C  = MovieList(unique([A.movieDataFile_ B.movieDataFile_]),outputDirectory);
end