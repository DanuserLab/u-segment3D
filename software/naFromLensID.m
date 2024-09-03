function [NA,mag] = naFromLensID(lensID,ask)
%NAFROMLENSID is a lookup table to find the NA for a given lensID
%
% SYNOPSIS  [NA,mag] = naFromLensID(lensID,ask)
%
% INPUT     lensID : ID of the lens according to deltaVision
%           ask    : Whether or not to ask for lensID if not found in list
%                    Default: 1
%
% OUTPUT    NA  : numerical aperture of the specified lens
%           mag : magnification of the lens
%
% REMARKS   After the program asks for a lensID, it will write it into the
%             .m-file. Alternatively, you can write it into the list at the
%             end of the code yourself.
%
% c: 8/05 jonas
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


% TEST INPUT
if nargin == 0
    error('not enough input arguments')
end
if nargin < 2 || isempty(ask)
    ask = 1;
end
NA = NaN;
mag = NaN;

% LOOKUP LENSID
lensIDList = lookupLensIDs;

lensIdx = find(lensIDList(:,1) == lensID);

% assign NA and mag if possible
if ~isempty(lensIdx)
    % use most recent version
    NA = lensIDList(lensIdx(end),2);
    mag = lensIDList(lensIdx(end),3);
end

% check whether we have all the output we need
if ~isnan(NA) && (~isnan(mag) || nargout < 2)
    % all is fine
elseif ask
    % if not good, check whether we can ask for input. Use what we have for
    % defaults
    answ = inputdlg(...
        {sprintf('Please input NA of lens %i',lensID),...
        sprintf('Please input magnification of lens %i',lensID)},...
        'Manual NA',1,...
        {num2str(NA),num2str(mag)});
    if ~isempty(answ)
        NA = str2double(answ{1});
        mag = str2double(answ{2});
    end
    
    % write to file
    
    file = which('naFromLensID.m');
    mFile = textread(file,'%s',...
        'delimiter','\n','whitespace','');
    % now add a line just befor the last line
    newLine = sprintf('%6i,  %1.2f,%4i;...',lensID,NA,mag);
    mFile{end+1} = mFile{end};
    mFile{end-1} = newLine;

    % finally, write the mFile
    fid = fopen(file,'w');
    for i=1:length(mFile)
        fprintf(fid,'%s\n',mFile{i});
    end
    fclose(fid);
else
    % simply return the default (NaN)
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function lensID = lookupLensIDs
% add lensIDs, NAs and mags by adding a line like:
% '9999, 1.25, 60;...'
% into the table below. Do not add any lines after the
% closing of the table '];'

lensID = ...
    [10002,  1.40, 100;...
    10006,  1.40, 100;...
    10612,  1.42,  60;...
    12003,  1.40, 100;...
    0,  1.40, NaN;...
    10005,  1.35, NaN;...
    10105,  1.40, NaN;...
    10403,  1.35, NaN;...
    10003,  1.35, NaN;...
    12005,  NaN, NaN;...
    10007,  1.40, 100;...
    99999, 1.4, 100;...
    10211,  1.40, 100;...
    10602,  1.40,  60;...
    ];
