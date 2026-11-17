import React from 'react'
import { User, displayName, Role } from '@acme/core'

type Props = { user: User; onSelect?: (id: string) => void }

export default function ({ user, onSelect }: Props) {
  const label = displayName(user)
  const handleClick = () => onSelect?.(user.id)
  return (
    <div className="card" onClick={handleClick}>
      {/* ghost() is a comment, not a call */}
      <span title='it is "quoted"'>{label}</span>
      Don't panic — ghostText() is text. Role: {user.role === Role.Admin ? 'admin' : 'member'}
    </div>
  )
}
