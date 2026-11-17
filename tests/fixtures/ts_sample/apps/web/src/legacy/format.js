const { displayName } = require('@acme/core')

function shout(u) {
  return displayName(u).toUpperCase()
}

function whisper(u) {
  return shout(u).toLowerCase()
}

function flush(cache) {
  cache.invalidate()
}

module.exports = { shout, whisper, flush }
