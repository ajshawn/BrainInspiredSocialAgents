-- meltingpot/lua/modules/tile_dump.lua
local args      = require 'common.args'
local class     = require 'common.class'
local tensor    = require 'system.tensor'
local component = require 'meltingpot.lua.modules.component'
local registry  = require 'meltingpot.lua.modules.component_registry'

local TileDump = class.Class(component.Component)

function TileDump:__init__(kwargs)
  kwargs = args.parse(kwargs, {
    {'name',   args.default('TileDump')},
    {'width',  args.numberType},
    {'height', args.numberType},
  })
  TileDump.Base.__init__(self, kwargs)
  self._w = kwargs.width
  self._h = kwargs.height

  --0 is preserved for ignored tiles
  -- map acorn to 30, floorAcorn to 31
  self._state_to_code = {
    apple       = 30,
    floorAcorn = 31,
  }
  -- now map all player to 10-23
  for i = 0, 13 do
    self._state_to_code['player' .. i] = 10 + i
  end

end

function TileDump:addObservations(tileSet, _world, observations)
  local sim = assert(self.gameObject.simulation,
                     "TileDump: no simulation attached")
  local w, h = self._w, self._h
  local size = w * h

  local fn = function()
    -- build a flat Lua array of length w*h
    local flat = {}
    for i = 1, size do flat[i] = 0 end

    for _, obj in pairs(sim:getAllGameObjects()) do
      local code = self._state_to_code[obj:getState()]
      if code then
        -- getPosition() returns e.g. {x, y}
        local pos = obj:getComponent('Transform'):getPosition()
        local x, y = pos[1] + 1, pos[2] + 1 -- TO pair with Lua's 1-based indexing
        -- compute 1D index
        local idx = (y - 1) * w + x
        flat[idx] = code
      end
    end

    -- wrap into a 1-D Int32Tensor; DMLab will reshape to (w,h)
    return tensor.Int32Tensor(flat)
  end

  observations[#observations+1] = {
    name  = 'WORLD.TILE_CODES',
    type  = 'tensor.Int32Tensor',
    shape = {w, h},
    func  = fn,
  }
end

registry.registerComponent('TileDump', TileDump)
return TileDump
