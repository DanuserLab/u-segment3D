function [ tf ] = insequence( x, start, stop , interval)
%insequence Checks to see if value is in the sequence start:interval:stop
%
% Roughly equivalent to ismember(x,start:interval:stop)
%
% This function does not handle negative intervals
% It is faster than ismember since for a sequence one can first check to
% see if x in the interval [start stop], and then see if (x-start)/interval
% is an integer.
%
% INPUT
% x - vector to query if in the sequence
% start - beginning of sequence
% stop  - end of sequence
% interval - (optional) interval between numbers in sequence
%            default: 1
%
% OUTPUT
% tf - logical vector same size as x
%
% See also ismember, insequence_and_scalar, all_insequence
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

% Mark Kittisopikul, April 2017
% Jaqaman Lab
% UT Southwestern

if(isscalar(x))

    if(x < start || x > stop)
        tf = false;
        return;
    end

    if(nargin < 4)
        x = (x - start);

    else
        x = (x - start)./interval;
    end
    tf = x == round(x);

else
    
    tf = x >= start & x <= stop;
    
    if(nargin < 4)
        x = x(tf) - start;
    else
        x = (x(tf) - start)./interval;
    end
    tf(tf) = x == round(x);
       
end

end

