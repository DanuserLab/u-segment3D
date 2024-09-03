function varargout = getLensProperties(id,varargin)
% GETLENSPROPERTIES retrieve the properties associated with a given lens
%
% GETLENSPROPERTIES first looks for a look-up table called dvlenses.tab. If
% not found, it switches to naFromLensID. The properties of the input lens
% are then parsed and returned as a variable number of output arguments.
%
%
% Input:
%    - id : a scalar giving the id of the lens.
%
%    - properties (optional): a cell array containing the properties to be
%    retrieved. 
%    Acceptable properties (as defined by dvlenses.tab): name, na, magn,wd,
%    fl, type
%    Default properties: na, magn.
%
% Output
%    - a variable number of properties depending on the size of the second
%    optional input.
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

% Sebastien Besson, Jan 2012

% Check input
ip=inputParser;
ip.addRequired('id',@isscalar);
validProps = {'name','magn','na','fl','wd','type'};
ip.addOptional('props',{'na','magn'},@(x) iscell(x) && all(ismember(x,validProps)));
ip.parse(id,varargin{:});
props=ip.Results.props;

% Check look-up table
lut_path = which('dvlenses.tab');
if isempty(lut_path), varargout=naFromLensID(id); end

% Get the full lens database for parsing
fid = fopen(lut_path);
lensDbText = fscanf(fid,'%c');
fclose(fid);

% Remove all lines preceding first lens description
start=regexp(lensDbText,'\<id=','start','once');
lensDbText=lensDbText(start:end);

% Split lens description into cell array and prune out invalid ids
lenses=regexp(lensDbText,'\n{2,}','split');
allIds=regexp(lenses,'\<id=(\d)+\>','tokens');
validLenses = ~cellfun(@isempty,allIds);
lenses=lenses(validLenses);
allIds=cellfun(@(x) str2double(x{1}),allIds(validLenses));
assert(numel(unique(allIds))==numel(allIds));

% Get full description of input lens
lensIndex=find(id==allIds,1);
if isempty(lensIndex), error('Lens Id not found in the database');end
lens = lenses{lensIndex};
   
% Numerical aperture
token =regexp(lens,'\<na=(\S+)\>','tokens');
assert(~isempty(token) && ~isnan(str2double(token{1})));
lensData.na= str2double(token{1});

% Magnification
token =regexp(lens,'\<magn=(\S+)\>','tokens');
assert(~isempty(token) && ~isnan(str2double(token{1})));
lensData.magn= str2double(token{1});

% Lens name (can contain whitespaces)
token =regexp(lens,'\<name=([\S ]+)\n','tokens');
assert(~isempty(token));
lensData.name= token{1};

% Lens type
token =regexp(lens,'\<type=(\S+)\>','tokens');
if ~isempty(token), lensData.type= token{1}; else lensData.type=''; end

% Focal length
token =regexp(lens,'\<fl=(\S+)\>','tokens');
if ~isempty(token), 
    lensData.fl= str2double(token{1}); 
else 
    lensData.fl= NaN; 
end

% Working distance
token =regexp(lens,'\<wd=(\S+)\>','tokens');
if ~isempty(token)
    lensData.wd= str2double(token{1});
else
    lensData.wd= NaN;
end

% Return queried properties
for i=1:numel(props), varargout{i}=lensData.(props{i}); end