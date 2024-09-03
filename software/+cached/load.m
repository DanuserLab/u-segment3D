function [ S , cached] = load( filename, varargin )
% function [ S , cached] = cached.load( filename, '-cacheFlag', '-loadFlag', variables )
%cached.load Emulates the built-in load but adds a caching facility so that
%subsequent loads read from memory rather than disk
%
% cached.load resolves the location of the MAT file indicated by filename
% and stores the contents in memory. The next request to load that MAT file
% is loaded from memory rather than the disk. 
%
% If any variables are indicated, then the cache is checked for those
% variables and loaded if necessary. This is highly recommended to
% ensure the needed variables are loaded.
%
% If all variables contained in the file are needed and variables are not
% listed, it is recommended that the -reset flag is used to ensure all
% variables are loaded at least once. Otherwise, only the previously
% requested variables may be returned. While this will not cache the
% current loading operation, it may assist subsequent loads.
%
% This function is faster if filename is an absolute path.
%
% Usage
% -----
% S = cached.load( ____ );
%     Just like the built-in load, caches the output
% S = cached.load(filename, '-reset', ____ )
%     resets the cache for filename
% S = cached.load(filename, '-useCache', false , ____ )
%     resets the cache for filename
% S = cached.load(filename, '-useCache', true , ____ )
%     same as cached.load(filename, _____)
% cached.load('-clear') 
%     clears the entire cache
%
% See also load
% 
% Output
% ------
% S is the structure output by load or empty
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
% 4. Remove variables in the matfile that were not requested
% 5. If no output is requested, assign values in the caller like load

persistent cache;

% reset is false by default
reset = false;
clearKey = false;
% S is empty as a flag if successfully loaded the cache
S = [];
cached = false;

%% 1. Process arguments

% First argument can be an optional argument to clear the entire cache
if(strcmp(filename,'-clear'))
    delete(cache);
    return;
end

forwarded = varargin;

for ii=1:length(varargin)
    % break out of the loop once we find one of these exclusive options
    switch(varargin{ii})
        case '-reset'
        % Second argument can be an optional reset flag which will clear the cache
        % corresponding to the other arguments
        
        % remove the -reset option since we do not want to forward it to load
            forwarded = varargin([1:ii-1 ii+1:end]);
            reset = true;
            break;
        case '-useCache'
        % Second argument can also be a -useCache parameter followed by a boolean
            assert(islogical(varargin{ii+1}),'-useCache must be followed by a logical');
            % if useCache == true, then do not reset. Reset otherwise.
            reset = ~varargin{ii+1};
            % remove the name/value since we do not want to forward it to load
            forwarded = varargin([1:ii-1 ii+2:end]);
            break;
        case '-clear'
        % The -clear option with a filename will clear that specific
        % cache
            clearKey = true;
            forwarded = varargin([1:ii-1 ii+1:end]);
            break;
    end
end



% Create the cache if it has not been created
if(~isa(cache,'containers.Map') || ~isvalid(cache))
    % If we are using a version less than 2008b, then just load and do not
    % cache since containers.Map was not introduced until 2008b
    if(verLessThan('matlab','7.7'))
        S = load(filename,forwarded{:});
        return;
    else
        cache = containers.Map;
    end
end

% Options begin with a dash
isOption = strncmp(forwarded,'-',1);
options = forwarded(isOption);
% Variables do not begin with a dash
variables = forwarded(~isOption);

% Resolve the full file location
if(~cache.isKey(filename))
    filename = whichMatFile(filename);
end

% The Map key is the filename and options joined together
key = filename;
if(~isempty(options))
    key = strjoin( [filename options], ',');
end

if((reset || clearKey) && cache.isKey(key))
    cache.remove(key);
end

if(clearKey)
    return;
end

D = dir(filename);

%% 2. Attempt to retrieve from the cache
if(cache.isKey(key))
    [S, R] = getData(key);
    % The cached data is valid if
    % 1. The modification times are the same AND
    % 2a. Variables to retrieve were specified OR
    % 2b. All variables were loaded when cached

    if(R.last_modified == D.datenum && R.bytes == D.bytes)
        cached = true;
        if(~isempty(variables))
            if(any(~isfield(S,variables)))
                % if any of the requested variables are not present,
                % then update the cache
                S_fields = fieldnames(S);
                newVariables = setdiff(variables,S_fields);
                newS = load(filename,options{:},newVariables{:});
                newS_fields = fieldnames(newS);
                % NB: Documented tip in struct2cell:
                % Order is preserved between fieldnames and struct2cell
                S = cell2struct([ struct2cell(S) ; struct2cell(newS)], ...
                                [ S_fields       ; newS_fields]      , 1);
                % all variables were not loaded
                setData(key,S,false,D.datenum,D.bytes);
            end
        elseif(~R.all_variables)
            % 2. No variables were specified AND all variables were not cached
            S = [];
        end
    else
        % 1. The modification time is different 
        S = [];
    end
end

%% 3. Cache is invalid so load it normally
if(isempty(S))
    S = load(filename,forwarded{:});
    % all variables will be loaded 
    setData(key, S, isempty(variables), D.datenum, D.bytes);
    cached = false;
end

%% 4. Limit output structure to only the variables requested
if(~isempty(variables))
    bigS = S;
    S = struct();
    for f = 1:length(variables)
        S.(variables{f}) = bigS.(variables{f});
    end
end

%% 5. If no output is requested, assign values in the caller like builtin
if(nargout == 0)
    % if we loaded in ascii mode, then S may not be a struct
    if(isstruct(S))
        S_fields = fieldnames(S);
        for f = 1:length(S_fields)
            assignin('caller',S_fields{f},S.(S_fields{f}));
        end
    end
end

    function [S, record ] = getData(key)
        record = cache(key);
        S = record.data;
    end
    
    function setData(key,S,all_variables,last_modified,bytes)
        record.data = S;
        record.all_variables = all_variables;
        record.last_modified = last_modified;
        record.bytes = bytes;
        cache(key) = record;
    end
end

