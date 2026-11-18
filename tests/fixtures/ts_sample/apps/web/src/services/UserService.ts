import { Injectable, inject } from '@angular/core'
import { UserRepository } from '@acme/core'
import { Logger, Log } from './Logger'

@Injectable({ providedIn: 'root' })
export class UserService {
  private repo = inject(UserRepository)

  constructor(private logger: Logger) {}

  refresh(id: string) {
    this.logger.info(Log.line('refresh'))
    track('refresh')
    return this.repo.load(id)
  }
}
