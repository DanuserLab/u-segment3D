classdef FileCache < handle
% FileCache implements in an in-memory copy of the state of a file on the disk
%
% This class is meant to be a generic superclass to implementing caches based on files.
% The cached data becomes invalid if the file is modified.
%
% The keys in the internal cache are mapped to the _absolute_ filepaths.
%
% % Example: A simple cache
% myCache = cached.FileCache
% myCache.store('test.nd2') = 5;
% a = myCache.retrieve('test.nd2');
% % a = 5
% myCache.clear('test.nd2');
% a = myCache.retrieve('test.nd2')
% % a = []
%
% % Example: A simple cache for imfinfo
% myCache = cached.FileCache(@imfinfo);
% % runs imfinfo on test.tif
% myCache.retrieve('test.tif');
%
% See also cached.FileCacheWithIndexing, cached.MatFileCache, cached.load, cached.save
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
    properties
        % A containers.Map instance representing the backend of the cache
        cache
        % A function to invoke on a file if the cache state is stale or not loaded
        % The function should take a single parameter, key, which should be the
        % absolute fileName. The first output is the data to store. The optional
        % second output is a struct containing any extra record fields
        invalidCacheFunc
    end
    methods
        function obj = FileCache(invalidCacheFunc)
            % FileCache takes a single optional argument: invalidCacheFunc. 
            % See also cached.FileCache.invalidCacheFunc
            obj.cache = containers.Map;
            if(nargin > 0)
                N = nargout(invalidCacheFunc);
                if(N < 2 && N >= 0)
                    % N less than zero means function uses deal or
                    % varargout
                    obj.invalidCacheFunc = @(key) deal(invalidCacheFunc(key),struct);
                else
                    obj.invalidCacheFunc = invalidCacheFunc;
                end
            else
                obj.invalidCacheFunc = @(key) deal([],struct);
            end
        end
        function valid_tf = isValid(obj,fileName)
            %isValid determines if fileName can be resolved and a record exists in the cache
            try
                key = obj.resolveKey(fileName);
                valid_tf = obj.isValid_(key);
            catch
                valid_tf = false;
            end
        end
        function clear(obj,fileName)
            % clear(fileName) will remove the record for fileName
            key = obj.resolveKey(fileName);
            obj.cache.remove(key);
        end
        function clearAll(obj)
            % clearAll removes all records in this cache
            delete(obj.cache);
            obj.cache = containers.Map;
        end
        function store(obj,varargin)
            % store(fileName,data, ...) will store data in the cache.
            % extra parameters will be added to the cache record
            ip = inputParser;
            ip.addRequired('fileName',@ischar);
            ip.addOptional('data',obj.invalidCacheFunc,@(x) true);
            ip.KeepUnmatched = true;
            ip.parse(varargin{:});
            fileName = ip.Results.fileName;
            data = ip.Results.data;
            key = obj.resolveKey(fileName);
            if(isa(data,'function_handle'))
                [data,record] = data(key);
            else
                record = ip.Unmatched;
            end
            obj.setData(key,data,record);
        end
        function [data, record, cached] = retrieve(obj,fileName)
            % [data, record, cached] = retrieve(fileName) will retrieve the cached data
            % record contains file attributes to track the cache state
            % cached is a logical value indicating if the value was retrieved from the cache
            key = obj.resolveKey(fileName);
            if(obj.isValid_(key))
                [data, record] = obj.getData(key);
                cached = true;
            else
                [data,record] = obj.invalidCacheFunc(key);
                record = obj.setData(key,data,record);
                cached = false;
            end
        end
    end
    methods (Access = protected)
        function key = resolveKey(obj,fileName)
        % resolveKey obtains the absolute path of a file which is the key for the cache map
            key = [];
            if(obj.cache.isKey(fileName))
                key = fileName;
                return;
            else
                [s,attrib] = fileattrib(fileName);
                if(s)
                    key = attrib.Name;
                end
            end
            assert(~isempty(key),'cached.FileCache.resolveKey:invalidFileName','Invalid fileName. Cannot resolve cache key.');
        end
        function valid_tf = isValid_(obj,key)
            % isValid_ is the internal implementation of isValid indicating if the file record matches the file state on disk
            if(~obj.cache.isKey(key))
                valid_tf = false;
                return;
            end
            storedRecord = obj.getRecord(key);
            currentRecord = obj.getFileAttributes(key);
            % A record is valid if the modification date and the file size match the stored record
            valid_tf = ...
                   storedRecord.last_modified == currentRecord.last_modified ...
                && storedRecord.bytes == currentRecord.bytes;
        end
        function recordStruct = getFileAttributes(~,key,S)
            % getFileAttributes builds the basic cache record for key. The argument S is a struct
            % containing additional record fields.
            D = dir(key);
            if(nargin > 2)
                recordStruct = S;
            end
            recordStruct.last_modified = D.datenum;
            recordStruct.bytes = D.bytes;
        end
        function [data, record] = getData(obj,key)
            % getData gets the data for key
            record = obj.getRecord(key);
            data = record.data;
        end
        function record = setData(obj,key,data,R)
            % setData set the data for key. R is an optional struct containing additional record fields.
            if(nargin < 4)
                R = struct;
            end
            record = obj.getFileAttributes(key,R);
            record.data = data;
            obj.setRecord(key,record);
        end
        function record = getRecord(obj,key)
            % getRecord retrieves the record from the underlying containers.Map
            record = obj.cache(key);
        end
        function setRecord(obj,key,record)
            % setRecord sets the record for the underlying containers.Map
            obj.cache(key) = record;
        end
    end
end
