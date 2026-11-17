const format = require('./format')
const { whisper } = require('./format')

exports.report = function (users) {
  return users.map(u => format.shout(u)).concat([whisper(users[0])])
}
