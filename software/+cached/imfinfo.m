function [ info, cached ] = imfinfo( varargin )
%cached.imfinfo Avoid redundant calls to the builtin imfinfo by using caching to store information about graphics
% cached.imfinfo(filename, fmt, [option, [true/false]])
%
% cached.imfinfo resolves the location of the file indicated by filename
% and stores the contents in memory. The next request to get file
% information will be loaded from memory.
%
% This function is faster if filename is an absolute path.
%
% Usage
% -----
% S = cached.imfinfo( ____ );
%     Just like the built-in imfinfo, caches the output
% S = cached.imfinfo( ____ , '-reset')
%     resets the cache for filename
% S = cached.imfinfo( ____ '-useCache', false)
%     resets the cache for filename
% S = cached.imfinfo( ____, '-useCache', true)
%     same as cached.imfinfo( ____ )
% cached.imfinfo('-clear') 
%     clears the entire cache
%
% See also imfinfo
% 
% Output
% ------
% info is the structure output by imfinfo or empty
% cached is a logical flag indicating if the cached was used
%
% Mark Kittisopikul
% December 2014
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

% Outline
% 1. Process arguments and check version
% 2. Attempt to load from the cache
% 3. If cache is invalid, then load the file

persistent cache;

ip = inputParser;
ip.addRequired('filename',@ischar);
if(nargin > 1 && varargin{2}(1) ~= '-')
    ip.addOptional('fmt',[],@(x) x(1) ~= '-');
end
ip.addOptional('option','',@(x) x(1) == '-');
ip.addOptional('optionFlag',true,@islogical);
ip.parse(varargin{:});

in = ip.Results;
forwarded = { in.filename };
if(isfield(in,'fmt'))
    forwarded = [forwarded in.fmt];
else
    in.fmt = [];
end

% reset is false by default
reset = false;
clearKey = false;
info = [];
cached = false;

%% 1. Process arguments

% First argument can be an optional argument to clear the entire cache
if(strcmp(in.filename,'-clear'))
    delete(cache);
    return;
end

switch(in.option)
    case '-reset'
    % Third argument can be an optional reset flag which will clear the cache
    % corresponding to the other arguments

        reset = true;
    case '-useCache'
    % Third argument can also be a -useCache parameter followed by a boolean
        % if useCache == true, then do not reset. Reset otherwise.
        reset = ~in.optionFlag;
    case '-clear'
    % The -clear option with a filename will clear that specific
    % cache
        clearKey = true;
end

if(~isa(cache,'containers.Map') || ~isvalid(cache))
    % If we are using a version less than 2008b, then just use builtin and do not
    % cache since containers.Map was not introduced until 2008b
    if(verLessThan('matlab','7.7'))
        info = imfinfo(forwarded{:});
        return;
    else
        cache = containers.Map;
    end
end

% Resolve the full file location
if(~cache.isKey(in.filename))
    [s,attrib] = fileattrib(in.filename);
    if(s)
        in.filename = attrib.Name;
    end
end

% The Map key is the filename and fmt joined together
key = in.filename;
if(~isempty(in.fmt))
    key = strjoin( {in.filename in.fmt}, '.');
end

if(reset || clearKey && cache.isKey(key))
    cache.remove(key);
end

if(clearKey)
    return;
end

D = dir(in.filename);

%% 2. Attempt to retrieve from the cache
if(cache.isKey(key))
    [info, R] = getData(key);
    % The cached data is valid if
    % The modification times are the same and the file size is the same

    if(R.last_modified == D.datenum && R.bytes == D.bytes)
        cached = true;
    else
        % 1. The modification time is different 
        info = [];
    end
end

%% 3. Cache is invalid so load it normally
if(isempty(info))
    info = imfinfo(forwarded{:});
    setData(key, info, D.datenum, D.bytes);
    cached = false;
end


    function [S, record ] = getData(key)
        record = cache(key);
        S = record.data;
    end
    
    function setData(key,S,last_modified,bytes)
        record.data = S;
        record.last_modified = last_modified;
        record.bytes = bytes;
        cache(key) = record;
    end

end

