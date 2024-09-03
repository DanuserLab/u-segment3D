function [path,body,no,ext]=getFilenameBody(fname,separator)
%GETFILENAMEBODY splits a filename into its body, number and 
% extension part
%
% SYNOPSIS [path,body,no,ext]=getFilenameBody(fname)
% 
% INPUT fname : filename; the following filename structure must be 
%                         preserved:
%                         - alphanumeric body
%                         - number before extension
%                         - extension separated
%       separator: (opt) string preceding the number, e.g. '_'. Default: ''
%
% OUTPUT path : string with the path, [] if non-existent
%        body : string with body, [] if non-existent
%        no   : string with the number, [] if non-existent
%        ext  : extension, [] if non-existent
%
% SAMPLE getFileNameBody('test_1.tif') returns
%        []
%        'test_',
%        '1',
%        '.tif'
%
%        getFileNameBody('test_1.tif','_') returns
%        []
%        'test',
%        '1',
%        '.tif'
%   
%        getFileNameBody('C:\mydir\test1.tif') returns
%        'C:\mydir'
%        'test',
%        '1',
%        '.tif'
%
% SEE ALSO fileparts
%
% original author unknown. regexp formulation implemented by jonas
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

% initialize
path = [];
body = [];
no = [];
ext = [];

if nargin < 2
    separator = '';
end

% search for extension
[path,name,ext] = fileparts(fname);

% in the name: search for numbers that are possibly preceeded by the
% separator
[start dummy dummy dummy tokens] = regexp(name, [separator '(\d+)$']);
if ~isempty(tokens)
    tokens = tokens{1};
    no = tokens{1};
    body = name(1:start-1);
else
    body = name;
end



% % search for letters in remainder
% idxs=length(name);
% while(uint8(name(idxs))>47 && uint8(name(idxs))<58)
%    idxs = idxs-1;
% end;
% 
% % check whether this index points to an actual number
% if( length(name) ~= idxs )
%    no = name((idxs+1):length(name));
%    if(isempty(str2num(no)))
%       no = [];
%       error('unsupported filename format entered');
%    else
%       body = name(1:idxs);
%    end;
% else
%    body = name;
% end;

   
      
